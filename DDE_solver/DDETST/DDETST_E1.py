import numpy as np

from DDE_solver.rkh_refactor import *

r, c = np.pi/np.sqrt(3) + 1/20, np.sqrt(3)/(2*np.pi) - 1/25
def f(t, y, x, z):
    return r*y*(1 - x - c*z)

def phi(t):
    return 2 + t

def phi_t(t):
    return 1

def alpha(t, y):
    return t - 1

beta = alpha

t_span = [0, 40]


methods = ['RKC3', 'RKC4', 'RKC5']
tolerances = [1e-2, 1e-4, 1e-6, 1e-8, 1e-10]


for Tol in tolerances:
    print('===========================================================')
    print(f'Tol = {Tol} \n')
    for method in methods:
        solution = solve_ndde(t_span, f, alpha, beta, phi, phi_t, method = method, Atol=Tol, Rtol=Tol)

        
        print(f'method = {method}')
        print('No analytical solution')
        print('steps: ', solution.steps)
        print('fails: ', solution.fails)
        print('feval: ', solution.feval)
        print('')
