import numpy as np

from DDE_solver.rkh_refactor import *


def f(t, y, x):
    y1, y2 = y
    x1, _ = x  # x = [y1(t - y2(t)), y2(t - y2(t))], but only x1 is used
    dy1 = -2 * x1
    dy2 = (abs(x1) - abs(y1)) / (1 + abs(x1))
    return [dy1, dy2]


def phi(t):
    return [1.0, 0.5]


def alpha(t, y):
    y1, y2 = y
    return t - y2


def real_sol(t):
    # No known analytical solution
    return [np.nan, np.nan]


t_span = [0, 40]


methods = ['RKC3', 'RKC4', 'RKC5']
tolerances = [1e-2, 1e-4, 1e-6, 1e-8, 1e-10]


for Tol in tolerances:
    print('===========================================================')
    print(f'Tol = {Tol} \n')
    for method in methods:
        solution = solve_dde(f, alpha, phi, t_span, method = method, Atol=Tol, Rtol=Tol)

        
        print(f'method = {method}')
        print('No analytical solution')
        print('steps: ', solution.steps)
        print('fails: ', solution.fails)
        print('feval: ', solution.feval)
        print('')
