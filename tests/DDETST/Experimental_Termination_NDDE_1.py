""" This is an experimental feature, still needs more development"""

import numpy as np

from DDE_solver.rkh_refactor import *


def f(t, y, x, z):
    return -y/2 - z


def phi(t):
    return 1 - t


def phi_t(t):
    return -1


def alpha(t, y):
    return (t - 0.5)*y**2


epsilon = 0
t_span = [0, 4]


methods = ['RKC3', 'RKC4', 'RKC5']
tolerances = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-8, 1e-10, 1e-12]

ref_sol = solve_ndde(t_span, f, alpha, alpha, phi, phi_t, method=RKC5, Atol=1e-14, Rtol=1e-14)
end_point = ref_sol.t[-1]


for Tol in tolerances:
    print('===========================================================')
    print(f'Tol = {Tol} \n')
    for method in methods:
        solution = solve_ndde(t_span, f, alpha, alpha, phi, phi_t, method=method, Atol=Tol, Rtol=Tol)
        print(f'method = {method}')
        print('steps: ', solution.steps)
        print('fails: ', solution.fails)
        print('feval: ', solution.feval)
        print('end point: ', solution.t[-1])
        print('end point diff: ', abs(solution.t[-1] - end_point))
        print('')

