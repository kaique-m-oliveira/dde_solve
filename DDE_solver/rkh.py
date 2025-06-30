import numpy as np
import scipy as sc
from scipy.optimize import root


def one_step_RK4(f, tn, yn, h, alpha, eta):
    K1 = f(tn, yn, eta(alpha(tn)))
    K2 = f(tn + 0.5 * h, yn + 0.5 * h * K1, eta(alpha(tn + 0.5 * h)))
    K3 = f(tn + 0.5 * h, yn + 0.5 * h * K2, eta(alpha(tn + 0.5 * h)))
    K4 = f(tn + h, yn + h * K3, eta(alpha(tn + h)))

    yn_plus = yn + h*(K1 / 6 + K2 / 3 + K3 / 3 + K4 / 6)
    return K1, K2, K3, K4, yn_plus


def one_step_interpolants(f, tn, yn, yn_plus, h, alpha, eta, K1, K2, K3, K4, theta1):
    K5 = f(tn + h, yn_plus, eta(tn + h - alpha(tn + h)))

    def eta_0(theta):
        t2, t3 = theta * theta, theta * theta * theta

        d1 = 2 * t3 - 3 * t2 + 1
        d2 = -2 * t3 + 3 * t2
        d3 = t3 - 2 * t2 + theta
        d4 = t3 - t2
        return d1 * yn + d2 * yn_plus + d3 * h * K1 + d4 * h * K5

    tt = tn + theta1 * h
    K6 = f(tt, eta_0(tt), eta(alpha(tt)))

    def eta_1(theta):
        t2, t3 = theta * theta, theta * theta * theta
        nom1, den1 = (theta - 1) ** 2, 2 * theta1 - 1

        d1 = nom1 * (-3 * t2 + 2 * den1 * theta + den1) / den1
        d2 = t2 * (3 * t2 - 4(theta1 + 1) * theta + 6 * theta1) / den1
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
        return d1 * yn + d2 * yn_plus + d3 * h * K1 * d4 * h * K5 + d5 * h * K6

    return eta_0, eta_1, K5, K6


def error_est_method(tn, h, yn, f, yn_plus, eta, eta_1, alpha, K1, K5):
    # Lobatto formula now for pi1 and pi2

    pi1, pi2 = (5 - np.sqrt(5)) / 10, (5 + np.sqrt(5)) / 10
    t_pi1, t_pi2 = tn + pi1 * h, tn + pi2 * h

    K7 = f(t_pi1, eta_1(t_pi1), eta(alpha(t_pi1)))
    K8 = f(t_pi2, eta_1(t_pi2), eta(alpha(t_pi2)))

    yn_plus_tilde = yn + h * (K1 / 12 + 5 * K7 / 12 + 5 * K8 / 12 + K5 / 12)
    return yn_plus_tilde, K7, K8


def discrete_local_error_satistied(yn_plus_tilde, yn_plus, h, omega_min, rho, TOL):
    discrete_local_error = np.linalg.norm(yn_plus_tilde - yn_plus) / h  # eq 7.3.4

    if discrete_local_error <= TOL:
        return True, h
    else:
        h = (
            max(
                omega_min, min(omega_max, rho * (TOL / discrete_local_error) ** (1 / 4))
            )
            * h
        )
        return False, h

def uniform_local_error_satisfied(h, K1, K2, K3, K4, K5, K6, TOL, theta1, omega_min, rho):

    max_uniform_difference = (
        h
        * (32 * abs(2 * theta1 - 1))
        * np.linalg.norm(
            ((2 * theta1 - 1) / theta1) * K1
            - (2 * K2 + 2 * K3 + K4)
            + (3 * theta1 - 2) * K5 / (theta1 - 1)
            + K6 / (theta1 * (theta1 - 1))
        )
    )

    uniform_local_error = h * max_uniform_difference

    if uniform_local_error <= TOL:
        return True, h
    else:
        h = max(omega_min, rho * (TOL / uniform_local_error) ** (1 / 5)) * h
        return False, h

def try_step_CRK( f, tn, yn, h, alpha, eta, TOL=1e-08, theta1=1 / 3, omega_min=0.5, omega_max=1.5, rho=0.1):

    K1, K2, K3, K4, yn_plus = one_step_RK4(f, tn, yn, h, alpha, eta)

    eta_0, eta_1, K5, K6 = one_step_interpolants( f, tn, yn, h, alpha, eta, K1, K2, K3, K4)

    yn_plus_tilde, K7, K8 = error_est_method( tn, h, yn, f, yn_plus, eta, eta_1, alpha, K1, K5)

    _discrete_local_error_satistied, h = discrete_local_error_satistied(yn_plus_tilde, yn_plus, h, omega_min, rho, TOL):

    if not _discrete_local_error_satistied:
        return tn, None, None, h

    _uniform_local_error_satisfied, h = uniform_local_error_satisfied(h, K1, K2, K3, K4, K5, K6, TOL, theta1, omega_min, rho)

   if not _uniform_local_error_satisfied:
        return tn, None, None, h


    h_next = max( omega_min, min( omega_max, rho * (TOL / discrete_local_error) ** (1 / 4), rho * (TOL / uniform_local_error) ** (1 / 5))) * h

    return tn + h, yn_plus, eta_1, h_next




#TODO: gotta do the max iterations here and add the disc search as well, before doig DDE_solve
def one_step_CRK(f, tn, yn, h, alpha, eta, TOL=1e-08, theta1=1 / 3, max_rejected_steps=30, omega_min=0.5, omega_max=1.5, rho=0.1):
    tn_plus, yn_plus, eta_1, h_next = try_step_CRK( f, tn, yn, h, alpha, eta, TOL=TOL, theta1=theta1, omega_min=omega_min, omega_max=omega_max, rho=rho)

    if yn_plus == None:
        possible_disc = get_disc()

#TODO: at lot of work left
def DDE_solve(f, t_span, phi, alpha, TOL=10**-8, C=0.1):
    t0, tf = t_span
    t = t0
    h = (TOL ** (1 / 4)) / C
    while t <= tf:
        t_next = t + h

        t, y, h, eta_1 = try_step_CRK(f, tn, yn, h, alpha, eta, TOL=TOL)
        if y == None:
        





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
