import numpy as np

from DDE_solver.rkh_refactor import *


def f(t, y, x):
    return -2 * x * (1 - y**2)


def phi(t):
    return 0.5


def alpha(t, y):
    return t - 1 - abs(y)

t_span = [0, 30]

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
