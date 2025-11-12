import matplotlib.pyplot as plt
import numpy as np

from DDE_solver.rkh_refactor import *


def f(t, y, x):
    return -2 * x * (1 - y**2)


def phi(t):
    return 0.5


def alpha(t, y):
    return t - 1 - np.abs(y)

t_span = [0, 30]

methods = ['RKC3', 'RKC4', 'RKC5']
tolerances = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-8, 1e-10, 1e-12]


for Tol in tolerances:
    print('===========================================================')
    print(f'Tol = {Tol} \n')
    for method in methods:
        solution = solve_dde(t_span, f, alpha, phi, method = method, Atol=Tol, Rtol=Tol)
        
        diff = abs(0.99553206380499 - solution.eta(30))
        print(f'method = {method}')
        print('steps: ', solution.steps)
        print('fails: ', solution.fails)
        print('feval: ', solution.feval)
        print('diff at 30: ', diff)
        print('discs: ', solution.discs)
        print('')
