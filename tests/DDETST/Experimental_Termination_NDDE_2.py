""" This is an experimental feature, still needs more development"""

import numpy as np

from DDE_solver.rkh_refactor import *


def f(t, y, x, z):
    return np.cos(t)*(1 + x) + 0.6 * y* z


def phi(t):
    return -t/2


def phi_t(t):
    return -1/2


def alpha(t, y):
    return t*y**2


t_span = [0.25, 5]


methods = ['RKC3', 'RKC4', 'RKC5']
tolerances = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-8, 1e-10, 1e-12]



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
        print('')

