import numpy as np
from DDE_solver.rkh_refactor import *


def H(t):
    return np.where(t < 0, 0.0, 1.0)

def alpha(t, y):
    return 0.5 * t  

def beta(t, y):
    return t - np.pi  

def phi(t):
    return 0.0  

def phi_t(t):
    return 0.0  

def f(t, y, x, z):
    """
    y'(t) = 1 - 2*x**2
            - (1 + cos(t)) * H(t - pi) * (1 - 2*x**2)
            - (1 + cos(t)) * z
    where x = y(alpha(t, y)) and z = y'(beta(t, y))
    """
    return 1 - 2 * x**2 - (1 + np.cos(t)) * H(t - np.pi) * (1 - 2 * x**2) - (1 + np.cos(t)) * z

def real_sol(t):
    return np.sin(t)

t_span = [0.0, 4 * np.pi]


methods = ['CERK3', 'CERK4', 'CERK5']
tolerances = [1e-2,  1e-4, 1e-6, 1e-8, 1e-10]


for Tol in tolerances:
    print('===========================================================')
    print(f'Tol = {Tol} \n')
    for method in methods:
        solution = solve_ndde(t_span, f, alpha, beta, phi, phi_t, method = method, Atol=Tol, Rtol=Tol)

        max_diff = 0
        for i in range(len(solution.t) - 1):
            tt = np.linspace(solution.t[i], solution.t[i + 1], 100)
            sol = np.array([solution.eta(i) for i in tt])
            realsol = np.array([real_sol(i) for i in tt])
            max_diff_ = np.max(np.abs(realsol - sol))
            if max_diff_ > max_diff:
                max_diff = max_diff_
        
        print(f'method = {method}')
        print('max diff', max_diff)
        print('steps: ', solution.steps)
        print('fails: ', solution.fails)
        print('feval: ', solution.feval)
        print('discs: ', solution.discs)
        print('')
