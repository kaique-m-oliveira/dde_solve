import numpy as np

from dde_solve import *


def f(t, y, x):
    y1, y2 = y
    x1, x2 = x
    dy1 = 1.1/(1 + np.sqrt(10)*x1**(5/4)) - 10*y1/(1 + 40*y2)
    dy2 =100*y1/(1+40*y2) - 2.43*y2
    return dy1, dy2

def phi(t):
    return [1.05767027/3, 1.030713491/3]

def alpha(t, y):
    return t - 20


t_span = [0, 100]


methods = ['RKC3', 'RKC4', 'RKC5']
tolerances = [1e-2, 1e-3,  1e-4, 1e-6, 1e-8, 1e-10]


for Tol in tolerances:
    print(f'=====================================================') 
    print(f'Tol = {Tol} \n')
    for method in methods:
        solution = solve_dde(t_span, f, alpha, phi, method = method, Atol=Tol, Rtol=Tol)
        print(f'method = {method}')
        print('No analytical solution')
        print('steps: ', solution.steps)
        print('fails: ', solution.fails)
        print('feval: ', solution.feval)
        print('discs: ', solution.discs)
        print('')
