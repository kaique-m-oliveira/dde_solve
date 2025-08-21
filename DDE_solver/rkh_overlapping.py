import time
from bisect import bisect_right
from dataclasses import dataclass
import random
import matplotlib.pyplot as plt
import numpy as np
import scipy as sc
from scipy.optimize import root


@dataclass
class CRKParameters:
    theta1: float = 1 / 3
    TOL: float = 1e-7
    rho: float = 0.9
    omega_min: float = 0.5
    omega_max: float = 1.5


class OneStep:

    def __init__(self, solver, tn, h, yn):
        self.solver = solver
        self.h = h
        self.t = [tn, tn + self.h]
        self.h_next = None
        self.y = [yn, 1]  # we don't have yn_plus yet
        self.y_tilde = None
        self.K = np.array([None for i in range(8)])
        self.Y_tilde = np.zeros(8)
        self.eta = np.zeros(2)
        self.eta_t = np.zeros(2)
        self.disc_local_error = None
        self.uni_local_error = None
        self.params = CRKParameters()

    def _implicit_NCE(self):
        print('implicit')
        tn, h, yn = self.t[0], self.h, self.y[0]
        f, eta, alpha = self.solver.f, self.solver.eta, self.solver.alpha

        self.Y_tilde[0] = eta(alpha(tn, yn))
        self.K[0] = f(tn, yn, self.Y_tilde[0])
        K0 = self.K[0]  # explicitly known

        def F(K_guess):
            K1, K2, K3 = K_guess

            def eta_hat(t):
                theta = (t - tn) / h
                b1 = ((2/3) * theta**2 - (3/2) * theta + 1) * theta
                b2 = (-(2/3) * theta + 1) * theta**2
                b3 = (-(2/3) * theta + 1) * theta**2
                b4 = ((2/3) * theta - 0.5) * theta**2
                return yn + h * (b1*K0 + b2*K1 + b3*K2 + b4*K3)

            Y1 = eta_hat(alpha(tn + 0.5*h, yn + 0.5*h*K0))
            F1 = K1 - f(tn + 0.5*h, yn + 0.5*h*K0, Y1)

            Y2 = eta_hat(alpha(tn + 0.5*h, yn + 0.5*h*K1))
            F2 = K2 - f(tn + 0.5*h, yn + 0.5*h*K1, Y2)

            Y3 = eta_hat(alpha(tn + h, yn + h*K2))
            F3 = K3 - f(tn + h, yn + h*K2, Y3)

            return [F1, F2, F3]

        # Initial guess using K0
        K_init = [self.K[0], self.K[0], self.K[0]]

        sol = root(F, K_init, method='lm', tol=1e-14)
        if not sol.success:
            raise RuntimeError("Implicit NCE solver failed to converge.")

        self.K[1:4] = sol.x
        self.y[1] = self._eta_0_hat(tn + h)

    def _eta_0_hat(self, t):
        tn, h, yn = self.t[0], self.h, self.y[0]
        theta = (t - tn) / h
        b1 = ((2/3) * theta ** 2 - (3/2) * theta + 1) * theta
        b2 = (-(2/3) * theta + 1) * theta ** 2
        b3 = (-(2/3) * theta + 1) * theta ** 2
        b4 = ((2/3) * theta - (1/2)) * theta ** 2
        eta_0_hat = yn + h * (b1 * self.K[0] + b2 * self.K[1] +
                              b3 * self.K[2] + b4 * self.K[3])
        return eta_0_hat

    # WARN: Non full one
    def _Newton_simplified(self):
        tn, h, yn = self.t[0], self.h, self.y[0]
        f, eta, alpha = self.solver.f, self.solver.eta, self.solver.alpha
        I = np.eye(4)
        A = np.array([[0, 0, 0, 0], [1/2, 0, 0, 0],
                     [0, 1/2, 0, 0], [0, 0, 1, 0]])

        f_y, f_x, alpha_y, eta_t = self.solver.f_y, self.solver.f_x, self.solver.alpha_y, self.solver.eta_t
        J = I - h * np.kron(A, f_y + f_x * eta_t(alpha_n) * alpha_y)

    def one_step_RK4(self):
        tn, h, yn = self.t[0], self.h, self.y[0]
        f, eta, alpha = self.solver.f, self.solver.eta, self.solver.alpha
        self.Y_tilde[0] = eta(alpha(tn, yn))
        self.K[0] = f(tn, yn, self.Y_tilde[0])
        if alpha(tn + 0.5*h, yn + 0.5*h*self.K[0]) > tn:
            return self._implicit_NCE()
        self.Y_tilde[1] = eta(alpha(tn + 0.5*h, yn + 0.5*h*self.K[0]))
        self.K[1] = f(tn + 0.5 * h, yn + 0.5 * h * self.K[0], self.Y_tilde[1])
        if alpha(tn + 0.5*h, yn + 0.5*h*self.K[1]) > tn:
            return self._implicit_NCE()
        self.Y_tilde[2] = eta(alpha(tn + 0.5 * h, yn + 0.5*h*self.K[1]))
        self.K[2] = f(tn + 0.5 * h, yn + 0.5 * h * self.K[1], self.Y_tilde[2])
        if alpha(tn + h, yn + h*self.K[2]) > tn:
            return self._implicit_NCE()
        self.Y_tilde[3] = eta(alpha(tn + h, yn + h * self.K[2]))
        self.K[3] = f(tn + h, yn + h * self.K[2], self.Y_tilde[3])
        self.y[1] = yn + h * (self.K[0] / 6 + self.K[1] /
                              3 + self.K[2] / 3 + self.K[3] / 6)

    def _eta_0(self, t):
        tn, h, yn = self.t[0], self.h, self.y[0]
        theta = (t - tn) / h
        f, eta, alpha = self.solver.f, self.solver.eta, self.solver.alpha
        yn_plus = self.y[1]
        t2, t3 = theta**2, theta**3

        d1 = 2 * t3 - 3 * t2 + 1
        d2 = -2 * t3 + 3 * t2
        d3 = t3 - 2 * t2 + theta
        d4 = t3 - t2
        if alpha(tn + h, self.y[1]) > tn:
            self.K[4] = self._eta_0_hat(alpha(tn + h, self.y[1]))
        else:
            self.K[4] = f(tn + h, yn_plus, eta(alpha(tn + h, yn_plus)))
        eta0 = d1 * yn + d2 * yn_plus + d3 * h * self.K[0] + d4 * h * self.K[4]
        return eta0

    def _eta_1(self, t):
        theta1 = self.params.theta1
        theta = (t - self.t[0]) / self.h
        t2, t3 = theta * theta, theta * theta * theta
        nom1, den1 = (theta - 1) ** 2, 2 * theta1 - 1
        yn, yn_plus = self.y[0], self.y[1]
        h = self.h
        tn, yn = self.t[0], self.y[0]
        f, eta, alpha = self.solver.f, self.solver.eta, self.solver.alpha
        d1 = nom1 * (-3 * t2 + 2 * den1 * theta + den1) / den1
        d2 = t2 * (3 * t2 - 4 * (theta1 + 1) * theta + 6 * theta1) / den1
        d3 = (
            theta
            * nom1
            * ((1 - 3 * theta1) * theta + 2 * theta1 * den1)
            / (2 * theta1 * den1)
        )
        d4 = (
            t2
            * (theta - 1)
            * ((2 - 3 * theta1) * theta + theta1 * (4 * theta1 - 3))
            / (2 * (theta1 - 1) * den1)
        )
        d4 = t2 * nom1 / (2 * theta1 * den1 * (theta1 - 1))
        d5 = (1/(2 * theta1 * den1 * (theta1 - 1))) * t2 * nom1
        tt = tn + theta1 * h
        if alpha(tt, self.eta[0](tt)) > tn:
            self.Y_tilde[5] = self.eta[0](alpha(tt, self.eta[0](tt)))
        else:
            self.Y_tilde[5] = eta(alpha(tt, self.eta[0](tt)))
        self.K[5] = f(tt, self.eta[0](tt), self.Y_tilde[5])
        return (
            d1 * yn
            + d2 * yn_plus
            + d3 * h * self.K[0]
            + d4 * h * self.K[4]
            + d5 * h * self.K[5]
        )

    def _eta_1_t(self):
        # WARN: Como pegar os índices?

        return

    def error_est_method(self):
        # Lobatto formula now for pi1 and pi2
        tn, h, yn = self.t[0], self.h, self.y[0]
        f, eta, alpha = self.solver.f, self.solver.eta, self.solver.alpha

        pi1, pi2 = (5 - np.sqrt(5)) / 10, (5 + np.sqrt(5)) / 10
        t_pi1, t_pi2 = self.t[0] + pi1 * self.h, self.t[0] + pi2 * self.h
        if alpha(t_pi1, self.eta[1](t_pi1)) > tn:
            self.Y_tilde[6] = self.eta[1](alpha(t_pi1, self.eta[1](t_pi1)))
        else:
            self.Y_tilde[6] = eta(alpha(t_pi1, self.eta[1](t_pi1)))
        self.K[6] = f(t_pi1, self.eta[1](t_pi1), self.Y_tilde[6])
        if alpha(t_pi2, self.eta[1](t_pi2)) > tn:
            self.Y_tilde[7] = self.eta[1](alpha(t_pi2, self.eta[1](t_pi2)))
        else:
            self.Y_tilde[7] = eta(alpha(t_pi2, self.eta[1](t_pi2)))
        self.K[7] = f(t_pi2, self.eta[1](t_pi2), self.Y_tilde[7])
        K5 = f(tn + h, self.y[1], eta(alpha(tn + h, self.y[1])))

        self.y_tilde = self.y[0] + self.h * (
            (1/12)*self.K[0] + (5/12) * self.K[6] +
            (5/12) * self.K[7] + (1/12) * K5
        )

    def disc_local_error_satistied(self):

        self.disc_local_error = (
            np.linalg.norm(self.y_tilde - self.y[1]) / self.h
        )  # eq 7.3.4

        if self.disc_local_error <= self.params.TOL:
            return True
        else:
            self.h = self. h * (
                max(
                    self.params.omega_min,
                    min(
                        self.params.omega_max,
                        self.params.rho
                        * (self.params.TOL / self.disc_local_error) ** (1 / 4),
                    ),
                )
            )
            return False

    def uni_local_error_satistied(self):

        tn, h = self.t[0], self.h
        self.uni_local_error = (
            h * np.linalg.norm(self.eta[0](tn + (1/2) * h) - self.eta[1](tn + (1/2)*h)))
        # eq 7.3.4

        if self.uni_local_error <= self.params.TOL:
            return True
        else:
            self.h = self.h * (
                max(
                    self.params.omega_min, self.params.rho *
                    (self.params.TOL / self.disc_local_error) ** (1 / 5)
                )
            )
            return False

    def try_step_CRK(self):

        # eta = self.solver.eta
        # start_time = time.perf_counter()
        self.one_step_RK4()
        # K1, K2, K3, K4 = self.K[0], self.K[1], self.K[2], self.K[3]
        # self._implicit_NCE()
        # IK1, IK2, IK3, IK4 = self.K[0], self.K[1], self.K[2], self.K[3]
        # print(f'diff K1 {abs(K1 - IK1)}, K2 {abs(K2 - IK2)}, K3 {
        #       abs(K3 - IK3)}, K4 {abs(K4 - IK4)}')
        # end_time = time.perf_counter()
        # print('time for RK4', end_time - start_time)
        self.eta = [self._eta_0, self._eta_1]
        self.error_est_method()
        local_disc_satisfied = self.disc_local_error_satistied()
        uni_local_disc_satistied = self.uni_local_error_satistied()

        if not local_disc_satisfied:
            print(f'failed discrete t={self.t}, h={
                  self.h} error = {self.disc_local_error}')  # real diff = {abs(self.y[-1] - REAL_SOL(self.t[0] + self.h))}')
            # print(f't = {self.t[0] + self.h} \ny1 = {self.y[-1]},\nyt = {self.y_tilde},\nrs = {
            #       REAL_SOL(self.t[0] + self.h)}  ')
            return False

        if not uni_local_disc_satistied:
            print(f'failed uniform step {self.t} with h = {self.h}')
            return False

        # Handling divide by zero case
        if self.disc_local_error < 1e-14 or self.uni_local_error < 1e-14:
            self.h_next = self.params.omega_max * self.h
        else:
            self.h_next = self.h * max(
                self.params.omega_min,
                min(
                    self.params.omega_max,
                    self.params.rho * (self.params.TOL /
                                       self.disc_local_error)**(1/4),
                    self.params.rho * (self.params.TOL /
                                       self.uni_local_error)**(1/5)
                )
            )

        return True

    def one_step_CRK(self, max_iter=100):
        step_satisfied = self.try_step_CRK()

        if step_satisfied:
            return self
        else:
            for i in range(max_iter - 1):
                step_satisfied = self.try_step_CRK()
                if step_satisfied:
                    return self
            raise RuntimeError("one_step_CRK failed")

    def test_one_step_RK4(self):
        tn, h, yn = self.t[0], self.h, self.y[0]
        f, eta, alpha = self.solver.f, self.solver.eta, self.solver.alpha
        print('-'*80)
        print('test rk4')
        print('tn', tn)
        print('alpha atras', alpha(tn, yn), 'alpha exato', np.log(tn))
        # print('eta - sen', abs(tn - eta(alpha(tn, yn))))
        print('y1 - y(t1)', abs(tn + h - self.y[1]))
        print('K0, ..., K5', self.K[0], self.K[1],
              self.K[2], self.K[3], self.K[4], self.K[5])

        # return self.K[0], self.K[1], self.K[2], self.K[3], yn_plus

    def test_eta_0(self):
        tn, h, yn = self.t[0], self.h, self.y[0]
        f, eta, alpha = self.solver.f, self.solver.eta, self.solver.alpha
        print('-'*80)
        print('test_eta_0')
        print('tn', tn)
        # print('alpha atras', alpha(tn), 'alpha exato', (tn - np.pi/2))
        print('eta atras', eta(alpha(tn, yn)))
        print('eta - sen', abs(alpha(tn, yn) - eta(alpha(tn, yn))))
        print('y1 - sen(t1)', abs(tn + h - self.y[1]))
        tt = np.linspace(tn, tn + h, 100)
        sin = tt
        sol = np.array([self.eta[0](t) for t in tt])
        print("min", min(abs(sol - sin)), "max", max(abs(sol - sin)))
        print('K0, ..., K5', self.K[0], self.K[1],
              self.K[2], self.K[3], self.K[4], self.K[5])

    def test_disc_error_method(self):
        print('-'*80)
        print('testing error estimating method for example 1.3.1')
        def y(t): return 0 if t <= 1 else np.log(t)
        f, eta, alpha = self.solver.f, self.solver.eta, self.solver.alpha

        pi1, pi2 = (5 - np.sqrt(5)) / 10, (5 + np.sqrt(5)) / 10
        t_pi1, t_pi2 = self.t[0] + pi1 * self.h, self.t[0] + pi2 * self.h
        print(f'|Y_6 - y(alpha(t_pi1, y(t_pi1)))| {
              abs(self.Y_tilde[6] - y(alpha(t_pi1, np.log(t_pi1))))}')
        print(f't_pi1 {t_pi1} and t_pi2 {t_pi2}')
        print(f'|alpha(t_pi1, self.eta[1](t_pi1)) - alpha(t_pi1, y(t_pi1)| {
              alpha(t_pi1, self.eta[1](t_pi1)) - alpha(t_pi1, y(t_pi1))}')
        print(f'|alpha(t_pi2, self.eta[1](t_pi2)) - alpha(t_pi2, y(t_pi2)| {
              alpha(t_pi2, self.eta[1](t_pi2)) - alpha(t_pi2, y(t_pi2))}')
        print(f'|eta(alpha(t_pi1, self.eta[1](t_pi1))) - y(alpha(t_pi1, y(t_pi1))| {
              eta(alpha(t_pi1, self.eta[1](t_pi1))) - y(alpha(t_pi1, np.log(t_pi1)))}')
        print(f' |eta(0.3) - y(0.3)| {eta(0.3) - y(0.3)} ')
        yy = y(self.t[0] + self.h)
        print(f'|Y_7 - y(alpha(t_pi2, y(t_pi2)))| {
              abs(self.Y_tilde[7] - y(alpha(t_pi2, np.log(t_pi2))))}')
        print(
            f'|eta[1](t_pi1) - y(t_pi1)| {abs(self.eta[1](t_pi1) - y(t_pi1))}')
        print(
            f'|eta[1](t_pi2) - y(t_pi2)| {abs(self.eta[1](t_pi2) - y(t_pi2))}')
        print(f'|y_1 - y | {abs(self.y[1] - yy)}')
        print(f'|y_tilde - y | {abs(self.y_tilde - yy)}')
        print(f'we are at {self.t} ')
        # input('to proceed prees something')


class Solver:

    def __init__(self, f, alpha, phi, t_span):
        self.t_span = np.array(t_span)
        self.f = f
        self.f_y = None
        self.f_x = None
        self.alpha = alpha
        self.alpha_t = None
        self.alpha_y = None
        self.phi = phi
        self.phi_t = None
        self.t = [t_span[0]]
        self.steps = []
        self.y = [phi(t_span[0])]
        self.etas = [phi]
        self.etas_t = [self.phi_t]
        self.params = CRKParameters()

    # WARN: CHATGPT

    @ property
    def eta(self):
        def eval(t):
            idx = bisect_right(self.t, t)
            # Ensure t in [t_k-1, t_k] → use eta_k
            if idx == 0:
                return self.etas[0](t)
            if idx >= len(self.etas):
                # avoid evaluating past last known interval
                return self.etas[-1](t)
            if self.t[idx - 1] <= t <= self.t[idx]:
                return self.etas[idx](t)
            # Should not occur, but fallback safely
            return self.etas[max(0, idx - 1)](t)
        return eval

    @ property
    def eta_t(self):
        def eval(t):
            idx = bisect_right(self.t, t)
            # Ensure t in [t_k-1, t_k] → use eta_k
            if idx == 0:
                return self.etas_t[0](t)
            if idx >= len(self.etas_t):
                # avoid evaluating past last known interval
                return self.etas_t[-1](t)
            if self.t[idx - 1] <= t <= self.t[idx]:
                return self.etas_t[idx](t)
            # Should not occur, but fallback safely
            return self.etas_t[max(0, idx - 1)](t)
        return eval

    def solve_dde(self, discs=[], real_sol=None):
        # WARN: quick and dirty just for tests
        global REAL_SOL
        REAL_SOL = real_sol

        t, tf = self.t_span[0], self.t_span[-1]
        h = (self.params.TOL ** (1 / 4)) * 0.1  # Initial stepsize
        # print("-" * 80)
        # print("Initial h:", h)
        # print("-" * 80)

        step_counter = 0

        done = False
        while t < tf:
            # Ensure stepsize doesn't overshoot tf
            h = min(h, tf - t)

            onestep = OneStep(self, self.t[-1], h, self.y[-1])
            step = onestep.one_step_CRK()
            # print('counter', step_counter, 't', self.t[-1])
            step_counter += 1

            if step:  # Step accepted
                t = self.t[-1] + step.h
                self.t.append(t)
                self.steps.append(step)
                self.y.append(step.y[1])
                self.etas.append(step.eta[1])
                self.etas_t.append(step.eta_t)
                # if real_sol != None:
                #     print('accepted')
                #     print(f't = {t} \ny1 = {self.y[-1]},\nrs = {
                #           REAL_SOL(t)},\ndiff = {abs(self.y[-1] - real_sol(t))} h = {h}')
                # input('-'*80)
                h = onestep.h_next  # Use adjusted stepsize from rejection

            else:
                raise RuntimeError("one_step_CRK failed")

        # Final step to reach tf exactly
        if abs(t - tf) > 1e-10:  # Avoid numerical precision issues
            h = tf - self.t[-1]
            if h > 0:
                onestep = OneStep(self, self.t[-1], h, self.y[-1])
                step = onestep.one_step_CRK()
                if step:
                    self.t.append(tf)
                    self.steps.append(step)
                    self.y.append(step.y[1])
                    self.etas.append(step.eta[1])
                else:
                    print("Final step rejected, solution may be incomplete")
        print(step_counter)
