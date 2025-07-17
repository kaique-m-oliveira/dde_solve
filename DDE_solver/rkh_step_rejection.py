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
    TOL: float = 1e-8
    rho: float = 0.9
    omega_min: float = 0.5
    omega_max: float = 1.5


class OneStep:

    def __init__(self, solver, tn, h, yn):
        self.solver = solver
        self.t = [tn, tn + h]
        self.h = h
        self.h_next = None
        self.y = [yn, 1]  # we don't have yn_plus yet
        self.y_tilde = None
        self.K = np.zeros(8)
        self.Y_tilde = np.zeros(8)
        self.eta = np.zeros(2)
        self.disc_local_error = None
        self.uni_local_error = None
        self.params = CRKParameters()

    def one_step_RK4(self):
        tn, h, yn = self.t[0], self.h, self.y[0]
        f, eta, alpha = self.solver.f, self.solver.eta, self.solver.alpha
        Y_tilde1 = eta(alpha(tn, yn))
        K1 = f(tn, yn, Y_tilde1)
        Y_tilde2 = eta(alpha(tn + 0.5*h, yn + 0.5*h*K1))
        K2 = f(tn + 0.5 * h, yn + 0.5 * h * K1, Y_tilde2)
        Y_tilde3 = eta(alpha(tn + 0.5 * h, yn + 0.5*h*K2))
        K3 = f(tn + 0.5 * h, yn + 0.5 * h * K2, Y_tilde3)
        Y_tilde4 = eta(alpha(tn + h, yn + h * K3))
        K4 = f(tn + h, yn + h * K3, Y_tilde4)
        self.K[0:4] = [K1, K2, K3, K4]
        self.Y_tilde[0:4] = [Y_tilde1, Y_tilde2, Y_tilde3, Y_tilde4]
        self.y[1] = yn + h * (self.K[0] / 6 + self.K[1] /
                              3 + self.K[2] / 3 + self.K[3] / 6)

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

    def _eta_0(self, theta):
        tn, h, yn = self.t[0], self.t[1] - self.t[0], self.y[0]
        theta = (theta - tn) / h
        f, eta, alpha = self.solver.f, self.solver.eta, self.solver.alpha
        yn_plus = self.y[1]
        t2, t3 = theta**2, theta**3

        d1 = 2 * t3 - 3 * t2 + 1
        d2 = -2 * t3 + 3 * t2
        d3 = t3 - 2 * t2 + theta
        d4 = t3 - t2
        # print(
        #     f" tn {tn} \n h {h} \n alpha(tn + h) {alpha(tn + h)} "
        # )
        self.K[4] = f(tn + h, yn_plus, eta(alpha(tn + h, yn_plus)))
        eta0 = d1 * yn + d2 * yn_plus + d3 * h * self.K[0] + d4 * h * self.K[4]
        return eta0

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

    def _eta_1(self, theta):
        theta1 = self.params.theta1
        theta = (theta - self.t[0]) / self.h
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
        d5 = t2 * nom1 / (2 * theta1 * den1 * (theta1 - 1))
        tt = tn + theta1 * h
        self.Y_tilde[5] = eta(alpha(tt, self.eta[0](tt)))
        self.K[5] = f(tt, self.eta[0](tt), self.Y_tilde[5])
        return (
            d1 * yn
            + d2 * yn_plus
            + d3 * h * self.K[0]
            + d4 * h * self.K[4]
            + d5 * h * self.K[5]
        )

    def test_eta_1(self):
        tn, h, yn = self.t[0], self.h, self.y[0]
        f, eta, alpha = self.solver.f, self.solver.eta, self.solver.alpha
        print('-'*80)
        print('test_eta_1')
        print('tn', tn)
        print('alpha atras', alpha(tn, yn), 'alpha exato', (tn - np.pi/2))
        print('eta atras', eta(alpha(tn, yn)))
        print('eta - sen', abs(alpha(tn, yn) - eta(alpha(tn, yn))))
        print('y1 - sen(t1)', abs(tn + h - self.y[1]))
        tt = np.linspace(tn, (tn + h)/2, 100)
        sin = tt
        sol = np.array([self.eta[1](t) for t in tt])
        print("min", min(abs(sol - sin)), "max", max(abs(sol - sin)))
        print('K0, ..., K5', self.K[0], self.K[1],
              self.K[2], self.K[3], self.K[4], self.K[5])

    def try_step_CRK(self):

        self.one_step_RK4()
        # WARN: adding eta_0 for both fuck it
        self.eta = [self._eta_0, self._eta_0]

        # self.test_one_step_RK4()
        # self.test_eta_0()
        # input('one_step taken?')

        self.h_next = self.h

        return True

    def one_step_CRK(self, max_iter=30):
        step_satisfied = self.try_step_CRK()

        # WARN:
        return self


class Solver:

    def __init__(self, f, alpha, phi, t_span):
        self.t_span = np.array(t_span)
        self.f = f
        self.alpha = alpha
        self.phi = phi
        self.t = [t_span[0]]
        self.steps = []
        self.y = [phi(t_span[0])]
        self.etas = [phi]
        self.params = CRKParameters()

    # WARN: CHATGPT

    @property
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

    def solve_dde(self, discs=[]):
        t, tf = self.t_span[0], self.t_span[-1]
        h = (self.params.TOL ** (1 / 4)) * 0.1  # Initial stepsize
        print("-" * 80)
        print("Initial h:", h)
        print("-" * 80)

        done = False
        while t < tf:
            # WARN: varying h for test
            h = random.choice([0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3])*0.001

            # Ensure stepsize doesn't overshoot tf
            h = min(h, tf - t)

            # WARN: adding the discontinuity by force
            if done:
                h = 0.001
                done = False

            if len(discs) != 0:
                if discs[0] - t < 1.5*h and not done:
                    h = discs[0] - t
                    done = True
                    discs.pop(0)

            onestep = OneStep(self, self.t[-1], h, self.y[-1])
            step = onestep.one_step_CRK()

            if step:  # Step accepted
                t = self.t[-1] + step.h
                self.t.append(t)
                self.steps.append(step)
                self.y.append(step.y[1])
                self.etas.append(step.eta[1])

            else:
                h = onestep.h  # Use adjusted stepsize from rejection
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
