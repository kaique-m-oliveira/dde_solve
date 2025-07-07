from bisect import bisect_right
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import scipy as sc
from scipy.optimize import root


class OneStep:

    def __init__(self, solver, tn, h, yn):
        self.solver = solver
        self.t = [tn, tn + h]
        self.h = h
        self.h_next = None
        self.y = [yn, 0]  # we don't have yn_plus yet
        self.y_tilde = None
        self.K = np.zeros(8)
        self.eta = np.zeros(2)
        self.disc_local_error = None
        self.uni_local_error = None
        self.params = CRKParameters()

    def one_step_RK4(self):
        tn, h, yn = self.t[0], self.h, self.y[0]
        f, eta, alpha = self.solver.f, self.solver.eta, self.solver.alpha
        self.K[0] = f(tn, yn, eta(alpha(tn)))
        self.K[1] = f(tn + 0.5 * h, yn + 0.5 * h * self.K[0], eta(alpha(tn + 0.5 * h)))
        self.K[2] = f(tn + 0.5 * h, yn + 0.5 * h * self.K[1], eta(alpha(tn + 0.5 * h)))
        self.K[3] = f(tn + h, yn + h * self.K[2], eta(alpha(tn + h)))
        self.y[1] = yn + h * (
            self.K[0] / 6 + self.K[1] / 3 + self.K[2] / 3 + self.K[3] / 6
        )
        # return self.K[0], self.K[1], self.K[2], self.K[3], yn_plus

    def _eta_0(self, theta):
        tn, h, yn = self.t[0], self.t[1] - self.t[0], self.y[0]
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
        self.K[4] = f(tn + h, yn_plus, eta(alpha(tn + h)))
        return d1 * yn + d2 * yn_plus + d3 * h * self.K[0] + d4 * h * self.K[4]

    def _eta_1(self, theta):
        theta1 = self.params.theta1
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
        self.K[5] = f(tt, self.eta[0](tt), eta(alpha(tt)))
        return (
            d1 * yn
            + d2 * yn_plus
            + d3 * h * self.K[0]
            + d4 * h * self.K[4]
            + d5 * h * self.K[5]
        )

    def error_est_method(self):
        # Lobatto formula now for pi1 and pi2
        f, eta, alpha = self.solver.f, self.solver.eta, self.solver.alpha

        pi1, pi2 = (5 - np.sqrt(5)) / 10, (5 + np.sqrt(5)) / 10
        t_pi1, t_pi2 = self.t[0] + pi1 * self.h, self.t[0] + pi2 * self.h

        self.K[6] = f(t_pi1, self._eta_1(t_pi1), eta(alpha(t_pi1)))
        self.K[7] = f(t_pi2, self._eta_1(t_pi2), eta(alpha(t_pi2)))

        self.y_tilde = self.y[0] + self.h * (
            self.K[0] / 12 + 5 * self.K[6] / 12 + 5 * self.K[7] / 12 + self.K[4] / 12
        )

    def disc_local_error_satistied(self):

        self.disc_local_error = (
            np.linalg.norm(self.y_tilde - self.y[1]) / self.h
        )  # eq 7.3.4

        if self.disc_local_error <= self.params.TOL:
            return True
        else:
            self.h = (
                max(
                    self.params.omega_min,
                    min(
                        self.params.omega_max,
                        self.params.rho
                        * (self.params.TOL / self.disc_local_error) ** (1 / 4),
                    ),
                )
                * self.h
            )
            return False, self.disc_local_error

    def uni_local_error_satisfied(self):

        max_uni_difference = (
            self.h
            * (32 * abs(2 * self.params.theta1 - 1))
            * np.linalg.norm(
                ((2 * self.params.theta1 - 1) / self.params.theta1) * self.K[0]
                - (2 * self.K[1] + 2 * self.K[2] + self.K[3])
                + (3 * self.params.theta1 - 2) * self.K[4] / (self.params.theta1 - 1)
                + self.K[5] / (self.params.theta1 * (self.params.theta1 - 1))
            )
        )

        self.uni_local_error = self.h * max_uni_difference

        if self.uni_local_error <= self.params.TOL:
            return True
        else:
            self.h = (
                max(
                    self.params.omega_min,
                    self.params.rho
                    * (self.params.TOL / self.uni_local_error) ** (1 / 5),
                )
                * self.h
            )
            print("oops")
            return False, self.uni_local_error

    def try_step_CRK(self):

        self.one_step_RK4()
        self.eta = [self._eta_0, self._eta_1]
        self.error_est_method()

        _disc_local_error_satistied = self.disc_local_error_satistied()

        if not _disc_local_error_satistied:
            return False

        _uni_local_error_satisfied = self.uni_local_error_satisfied()

        if not _uni_local_error_satisfied:
            return False

        self.h_next = (
            max(
                self.params.omega_min,
                min(
                    self.params.omega_max,
                    self.params.rho
                    * (self.params.TOL / self.disc_local_error) ** (1 / 4),
                    self.params.rho
                    * (self.params.TOL / self.uni_local_error) ** (1 / 5),
                ),
            )
            * self.h
        )

        return True

    def one_step_CRK(self, max_iter=30):
        step_satisfied = self.try_step_CRK()

        # TODO: ADICIONAR A BUSCA POR DESCONTINUIDADE AQUI
        if step_satisfied:
            return self

        for i in range(max_iter):
            step_satisfied = self.try_step_CRK()
            if step_satisfied:
                return self

        raise RuntimeError(
            f"Max iterations reached for step at tn = {self.t} and h = {self.h}"
        )


@dataclass
class CRKParameters:
    theta1: float = 1 / 3
    TOL: float = 1e-8
    rho: float = 0.9
    omega_min: float = 0.5
    omega_max: float = 1.5


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

    # @property
    # def eta(self):
    #     def eval(t):
    #         for i in range(len(self.t)):
    #             if t <= self.t[i]:
    #                 # print("in, t < ti", t, self.t[i])
    #                 return self.etas[i](t)
    #             # print("out, t < ti", t, self.t[i])
    #
    #     # print("eval", eval)
    #     return eval

    @property
    def eta(self):
        def eval(t):
            idx = bisect_right(self.t, t)
            if idx == 0:
                return self.etas[0](t)
            if idx >= len(self.etas):
                return self.etas[-1](t)
            return self.etas[idx - 1](t)

        return eval

    def solve_dde(self):
        t, h, tf = self.t_span[0], (self.params.TOL ** (1 / 4)) * 0.1, self.t_span[-1]
        print("-" * 80)
        print("value h", h)
        print("-" * 80)
        onestep = OneStep(self, self.t[-1], h, self.y[-1])
        step = onestep.one_step_CRK()
        self.steps.append(step)
        while t < tf:
            onestep = OneStep(self, self.t[-1], h, self.y[-1])
            step = onestep.one_step_CRK()
            t += step.h_next
            print(t)
            if t < self.t_span[-1]:
                self.t.append(t)
                self.steps.append(step)
                self.y.append(step.y[1])
                self.etas.append(step.eta[1])
                h = step.h_next
            else:
                print("this is fucked")

        h = tf - self.t[-1]
        onestep = OneStep(self, tf, h, self.y[-1])
        step = onestep.one_step_CRK()
        self.t.append(tf)
        self.steps.append(step)
        self.y.append(step.y[1])
        self.etas.append(step.eta[1])


def f(t, y, yq):
    return -yq


def phi(t):
    return np.sin(t)


def alpha(t):
    return t - np.pi / 2


t_span = [0, 1.3]

solver = Solver(f, alpha, phi, t_span)
solver.solve_dde()

x = np.linspace(0, 1.3, 100)
sin = np.sin(x)
sol = np.array([solver.eta(i) for i in x])
print("min", min(abs(sol - sin)), "max", max(abs(sol - sin)))
#

plt.plot(x, sin, color="red")
plt.plot(x, sol, color="blue")
plt.show()


# class OneStepData:
#     def __init__(self, tn, h):
#         self.tn = tn
#         self.h = h
#         self.yn = None
#         self.yn_plus = None
#         self.K1 = None
#         self.K2 = None
#         self.K3 = None
#         self.K4 = None
#         self.K5 = None
#         self.K6 = None
#         self.K7 = None
#         self.K8 = None
#         self.eta0 = None
#         self.eta1 = None
#
#
# class ProblemData:
#
#     def __init__(self, t_span, f, alpha, phi):
#         self.t_span = t_span
#         self.f = f
#         self.alpha = alpha
#         self.phi = phi
#
#
# class Solution:
#     """initialize the solution with history data"""
#
#     def __init__(self, t0, phi):
#         self.t = [t0]
#         self.y = [phi(t0)]
#         self.eta = [phi]
#
#     def add_solution(self, t1, y1, eta1):
#         self.t.append(t1)
#         self.y.append(y1)
#         self.eta.append(eta1)
#
#     def get_eta(self, t):
#         for i in range(len(t)):
#             if t <= t[i]:
#                 return self.eta[i](t)
#
#
# def solve_DDE(t_span, f, alpha, phi):
#     problemData = ProblemData(t_span, f, alpha, phi)
#     solution = Solution(t_span[0], phi)
#
#
# # TODO: gotta convert all this to classes now
#
