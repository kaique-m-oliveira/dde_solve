""" This is an experimental feature, still needs more development"""

import numpy as np

from DDE_solver.rkh_refactor import *


def f(t, y, yq):
    return -yq + 5


def phi(t):
    return 9/2 if t < -1 else -1/2


def alpha(t, y):
    return t - 2 - y**2

def real_sol(t):
    if 0 <= t <= 1:
        return (1/2)*(t-1)
    elif 1 <= t <= 125/121:
        return (11/2)*(t-1)


t_span = [0, 2]
discs = [(-1, 9/2, -1/2)]

methods = ['RKC3', 'RKC4', 'RKC5']
tolerances = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-8, 1e-10, 1e-12]
end_point = 125/121


for Tol in tolerances:
    print('===========================================================')
    print(f'Tol = {Tol} \n')
    for method in methods:
        solution = solve_dde(t_span, f, alpha, phi, discs=discs, method = method, Atol=Tol, Rtol=Tol)

        
        print(f'method = {method}')
        print('steps: ', solution.steps)
        print('fails: ', solution.fails)
        print('feval: ', solution.feval)
        print('termination at: ', solution.t[-1])
        print('end_point diff: ', abs(solution.t[-1] - end_point))
        print('')
