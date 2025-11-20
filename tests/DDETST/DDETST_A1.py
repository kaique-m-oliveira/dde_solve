import numpy as np

from dde_solve import *


def f(t, y, x):
    return 0.2*x/(1+x**10) - 0.1*y

def phi(t):
    return 0.5

def alpha(t, y):
    return t - 14

# No analytical solution found 


t_span = [0, 500]


methods = ['CERK3', 'CERK4', 'CERK5']
tolerances = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-8, 1e-10]


for Tol in tolerances:
    print(f'=====================================================') 
    print(f'Tol = {Tol} \n')
    for method in methods:
        solution = solve_dde(t_span, f, alpha, phi, method = method, Atol=Tol, Rtol=Tol)
        print(f'method = {method}')
        print('steps: ', solution.steps)
        print('fails: ', solution.fails)
        print('feval: ', solution.feval)
        print('discs: ', solution.discs)
        print('')
