import time
from bisect import bisect_right
from dataclasses import dataclass, field
import random
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import lu_factor, lu_solve, norm
from scipy.optimize import root
from scipy.integrate import solve_ivp


@dataclass
class CRKParameters:
    theta1: float = 1 / 3
    TOL: float = 1e-1
    rho: float = 0.9
    omega_min: float = 0.5
    omega_max: float = 1.5
    A: np.ndarray = field(default_factory=lambda: np.array([
        [0,   0,   0, 0],
        [1/2, 0,   0, 0],
        [0,  1/2,  0, 0],
        [0,   0,   1, 0]
    ], dtype=float))
    b: np.ndarray = field(default_factory=lambda: np.array(
        [1/6, 1/3, 1/3, 1/6], dtype=float))
    c: np.ndarray = field(default_factory=lambda: np.array(
        [0, 1/2, 1/2, 1], dtype=float))


class OneStep:

    def __init__(self, solver, tn, h, yn):
        self.solver = solver
        self.h = h
        self.t = [tn, tn + self.h]
        self.h_next = None
        self.y = [yn, 1]  # we don't have yn_plus yet
        self.y_tilde = None
        self.K = np.zeros(8)
        self.eta = np.zeros(2)
        self.eta_t = np.zeros(2)
        self.disc_local_error = None
        self.uni_local_error = None
        self.params = CRKParameters()

    def one_step_RK4(self):
        tn, h, yn = self.t[0], self.h, self.y[0]
        f, alpha = self.solver.f, self.solver.alpha
        eta, eta_t = self.solver.eta, self.solver.eta_t
        c = self.params.c

        for i in range(4):
            if alpha(tn + c[i] * h, yn + c[i] * h) <= tn:
                Y_tilde = eta(
                    alpha(tn + c[i] * h, yn + c[i] * h * self.K[i - 1]))
                Z_tilde = eta_t(
                    alpha(tn + c[i] * h, yn + c[i] * h * self.K[i - 1]))
            else:  # this would be the overlapping case
                self._simplified_Newton()
                break
            self.K[i] = f(tn + c[i] * h, yn + c[i] * h *
                          self.K[i-1], Y_tilde, Z_tilde)

        self.y[1] = yn + h * (self.K[0] / 6 + self.K[1] /
                              3 + self.K[2] / 3 + self.K[3] / 6)

    def _simplified_Newton(self):
        print('Newton', self.t)
        I = np.eye(4)
        A, b, c = self.params.A, self.params.b, self.params.c
        rho, TOL = self.params.rho, self.params.TOL
        f_y, f_x = self.solver.f_y, self.solver.f_x
        eta_t, alpha_y = self.solver.eta_t, self.solver.alpha_y
        tn, h, yn = self.t[0], self.h, self.y[0]
        f, alpha = self.solver.f, self.solver.alpha
        eta, eta_t = self.solver.eta, self.solver.eta_t
        eta, eta_t = self.solver.eta, self.solver.eta_t
        yn_plus = self.y[1]
        # WARN: theta é uma aprox de theta_i, não sei outra forma de fazer isso
        alpha_n = alpha(tn, yn)
        theta = (alpha_n - tn) / h
        t2, t3 = theta**2, theta**3

        d1 = ((2/3) * t2 - (3/2) * theta + 1) * theta
        d2 = ((-2/3) * theta + 1) * t2
        d3 = ((2/3) * theta + 1) * t2
        d4 = ((2/3) * theta - 1/2) * t2

        B = np.array([[d3, d1, d1, d1], [d2, d2, d2, d2],
                     [d3, d3, d3, d3], [d4, d4, d4, d4]])

        # FIX: gotta make this check automatic
        if alpha_n <= tn:
            d_eta = eta_t
        else:
            d_eta = self._hat_eta_0_t

        J = I - h * np.kron(A, f_y + f_x * d_eta(alpha_n) *
                            alpha_y) - h * np.kron(B, f_x)
        lu, piv = lu_factor(J)

        def F(K):
            F = np.zeros(4)
            for i in range(4):
                ti = tn + c[i] * h
                yi = yn + c[i] * h * self.K[i-1]
                # FIX: gotta make this check automatic
                if alpha(ti, yi) <= tn:
                    eeta = eta
                    zeta = eta_t
                else:
                    eeta = self._hat_eta_0
                    zeta = self._hat_eta_0_t
                Y_tilde = eeta(alpha(ti, yi))
                Z_tilde = zeta(alpha(ti, yi))
                F[i] = K[i] - f(ti, yi, Y_tilde, Z_tilde)
            return F

        K = [i if i != 0 else self.K[0] for i in self.K[0:4]]
        max_iter, iter = 30, 0
        diff_old, diff_new = 4, 3  # initializing stuff
        while abs((norm(diff_new)**2)/(norm(diff_old) - norm(diff_new))) >= rho * TOL and iter <= max_iter:
            # Método de Newton usando recomposição LU
            diff_old = diff_new
            diff_new = lu_solve((lu, piv), - F(K))
            K += diff_new
            iter += 1
        self.K[0:4] = K

    def _eta_0(self, theta):
        tn, h, yn = self.t[0], self.h, self.y[0]
        theta = (theta - tn) / h
        f, alpha = self.solver.f, self.solver.alpha
        eta, eta_t = self.solver.eta, self.solver.eta_t
        yn_plus = self.y[1]
        t2, t3 = theta**2, theta**3

        d1 = 2 * t3 - 3 * t2 + 1
        d2 = -2 * t3 + 3 * t2
        d3 = t3 - 2 * t2 + theta
        d4 = t3 - t2
        if alpha(tn + h, yn_plus) <= tn:
            eeta = eta
            zeta = eta_t
        else:
            eeta = self._hat_eta_0
            zeta = self._hat_eta_0_t
        Y_tilde = eeta(alpha(tn + h, yn_plus))
        Z_tilde = zeta(alpha(tn + h, yn_plus))
        self.K[4] = f(tn + h, yn_plus, Y_tilde, Z_tilde)
        eta0 = d1 * yn + d2 * yn_plus + d3 * h * self.K[0] + d4 * h * self.K[4]
        return eta0

    def _hat_eta_0(self, theta):
        tn, h, yn = self.t[0], self.h, self.y[0]
        theta = (theta - tn) / h
        f, eta, alpha = self.solver.f, self.solver.eta, self.solver.alpha
        yn_plus = self.y[1]
        t2, t3 = theta**2, theta**3

        d1 = ((2/3) * t2 - (3/2) * theta + 1) * theta
        d2 = ((-2/3) * theta + 1) * t2
        d3 = ((2/3) * theta + 1) * t2
        d4 = ((2/3) * theta - 1/2) * t2
        eta0 = yn + h * (d1 * self.K[0] + d2 *
                         self.K[1] + d3 * self.K[2] + d4 * self.K[3])
        return eta0

    def _hat_eta_0_t(self, theta):
        tn, h, yn = self.t[0], self.h, self.y[0]
        theta = (theta - tn) / h
        f, eta, alpha = self.solver.f, self.solver.eta, self.solver.alpha
        yn_plus = self.y[1]

        d1 = 1 - 3 * theta + 2 * theta ** 2
        d2 = -2 * (-1 + theta) * theta
        d3 = d2
        d4 = theta * (-1 + 2 * theta)
        eta0 = h * (d1 * self.K[0] + d2 *
                    self.K[1] + d3 * self.K[2] + d4 * self.K[3])
        return eta0

    def _eta_1(self, theta):
        theta1 = self.params.theta1
        theta = (theta - self.t[0]) / self.h
        t2, t3 = theta * theta, theta * theta * theta
        nom1, den1 = (theta - 1) ** 2, 2 * theta1 - 1
        yn, yn_plus = self.y[0], self.y[1]
        h = self.h
        tn, yn = self.t[0], self.y[0]
        f, alpha = self.solver.f, self.solver.alpha
        eta, eta_t = self.solver.eta, self.solver.eta_t
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
        d5 = t2 * nom1 / (2 * theta1 * den1 * (theta1 - 1))
        tt = tn + theta1 * h

        if alpha(tt, self.eta[0](tt)) <= tn:
            eeta = eta
            zeta = eta_t
        else:
            eeta = self._hat_eta_0
            zeta = self._hat_eta_0_t
        Y_tilde = eeta(alpha(tt, self.eta[0](tt)))
        Z_tilde = zeta(alpha(tt, self.eta[0](tt)))
        self.K[5] = f(tt, self.eta[0](tt), Y_tilde, Z_tilde)
        return (
            d1 * yn
            + d2 * yn_plus
            + d3 * h * self.K[0]
            + d4 * h * self.K[4]
            + d5 * h * self.K[5]
        )

    def _eta_1_t(self, theta):
        theta1 = self.params.theta1
        theta = (theta - self.t[0]) / self.h
        t2, t3 = theta * theta, theta * theta * theta
        nom1, den1 = (theta - 1) ** 2, 2 * theta1 - 1
        yn, yn_plus = self.y[0], self.y[1]
        h = self.h
        tn, yn = self.t[0], self.y[0]
        f, eta, alpha = self.solver.f, self.solver.eta, self.solver.alpha
        x, a = theta, theta1

        d_d1 = (12*(a - x)*(-1 + x)*x)/(-1 + 2 * a)
        d_d2 = -(12 * (a - x)*(-1 + x) * x)/(-1 + 2 * a)
        d_d3 = ((-1 + x)*(a - 6 * a * x ** 2 + x * (-1 + 2 * x) +
                (a ** 2) * (-2 + 6*x)))/(a * (-1 + 2 * a))
        d_d4 = (x*(x*(-3 + 4 * x) + a ** 2) * (-4 + 6 * x) +
                a*(3 - 6 * (x ** 2)))/((-1 + a)*(-1 + 2 * a))
        d_d5 = (x - 3 * (x ** 2) + 2 * (x ** 3)) / \
            (a - 3 * (a ** 2) + 2 * (a ** 3))

        tt = tn + theta1 * h
        alpha_tt = alpha(tt, self.eta[1](tt))
        Y_tilde = eta(alpha_tt)
        self.K[5] = f(tt, self.eta[0](tt), Y_tilde)
        return (
            d_d1 * yn
            + d_d2 * yn_plus
            + d_d3 * h * self.K[0]
            + d_d4 * h * self.K[4]
            + d_d5 * h * self.K[5]
        )

    def error_est_method(self):
        # Lobatto formula now for pi1 and pi2
        f, alpha = self.solver.f, self.solver.alpha
        eta, eta_t = self.solver.eta, self.solver.eta_t

        pi1, pi2 = (5 - np.sqrt(5)) / 10, (5 + np.sqrt(5)) / 10
        t_pi1, t_pi2 = self.t[0] + pi1 * self.h, self.t[0] + pi2 * self.h
        Y_tilde6 = eta(alpha(t_pi1, self.eta[1](t_pi1)))
        Z_tilde6 = eta_t(alpha(t_pi1, self.eta[1](t_pi1)))
        self.K[6] = f(t_pi1, self.eta[1](t_pi1), Y_tilde6, Z_tilde6)
        Y_tilde7 = eta(alpha(t_pi2, self.eta[1](t_pi2)))
        Z_tilde7 = eta_t(alpha(t_pi2, self.eta[1](t_pi2)))
        self.K[7] = f(t_pi2, self.eta[1](t_pi2), Y_tilde7, Z_tilde7)

        self.y_tilde = self.y[0] + self.h * (
            (1/12)*self.K[0] + (5/12) * self.K[6] +
            (5/12) * self.K[7] + (1/12) * self.K[4]
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
            h * norm(self.eta[0](tn + (1/2) * h) - self.eta[1](tn + (1/2)*h)))
        # eq 7.3.4

        # print('disc val', self.disc_local_error)
        # print('uni val', self.uni_local_error)
        if self.disc_local_error <= self.params.TOL:
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
        time1 = time.time()
        self.one_step_RK4()
        time2 = time.time()
        self.eta = [self._eta_0, self._eta_1]
        time3 = time.time()
        self.eta_t = [self._eta_1_t, self._eta_1_t]
        time4 = time.time()
        self.error_est_method()
        time5 = time.time()
        # print(f'times: rk4 {time2 - time1}, eta {time3 -
        #       time2}, eta_t {time4 - time3}, err_est {time5 - time4}')
        local_disc_satisfied = self.disc_local_error_satistied()
        uni_local_disc_satistied = self.uni_local_error_satistied()

        if not local_disc_satisfied:
            print(f'failed discrete step at t = {self.t} with h = {self.h}')
            return False

        if not uni_local_disc_satistied:
            print(f'failed uniform step with h = {self.h}')
            return False
        # print(f'successfull step with h = {self.h}')
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
                                       self.disc_local_error)**(1/5)
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
            print('max iterations reached')


class Solver:

    def __init__(self, f, alpha, phi, t_span):
        self.t_span = np.array(t_span)
        self.f = f
        self.f_y = None
        self.f_x = None
        self.alpha = alpha
        self.alpha_t = None
        self.phi = phi
        self.phi_t = None
        self.t = [t_span[0]]
        self.steps = []
        self.y = [phi(t_span[0])]
        self.etas = [phi]
        self.etas_t = []
        self.params = CRKParameters()

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

    def solve_dde(self, discs=[]):
        t, tf = self.t_span[0], self.t_span[-1]
        h = (self.params.TOL ** (1 / 4)) * 0.1  # Initial stepsize
        # h = 0.1
        print("-" * 80)
        print("Initial h:", h)
        print("-" * 80)

        done = False
        while t < tf:
            # Ensure stepsize doesn't overshoot tf
            h = min(h, tf - t)

            # # WARN: varying h for test
            # h = random.choice([0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3])*0.001
            #
            # WARN: adding the discontinuity by force
            # if done:
            #     h = 0.001
            #     done = False
            #
            # if len(discs) != 0:
            #     if discs[0] - t < 1.5*h and not done:
            #         h = discs[0] - t
            #         done = True
            #         discs.pop(0)

            onestep = OneStep(self, self.t[-1], h, self.y[-1])
            step = onestep.one_step_CRK()

            if step:  # Step accepted
                t = self.t[-1] + step.h
                self.t.append(t)
                self.steps.append(step)
                self.y.append(step.y[1])
                self.etas.append(step.eta[1])

            else:
                h = onestep.h_next  # Use adjusted stepsize from rejection
                print(f"Step rejected at t={t:.6f}, new h={h:.6e}")

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
