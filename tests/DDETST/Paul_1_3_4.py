import numpy as np
from DDE_solver.rkh_refactor import *


def alpha(t, y):
    return np.log(y)

def phi(t):
    return 1.0

def f(t, y, x):
    return (y * x) / t

def real_sol(t):
    t = np.asarray(t)
    y = np.zeros_like(t)
    mask1 = (t >= 1) & (t <= np.e)
    mask2 = (t > np.e) & (t <= np.e**2)
    y[mask1] = t[mask1]
    y[mask2] = np.exp(t[mask2] / np.e)
    return y

t_span = [1.0, np.e**2]



methods = ['CERK3', 'CERK4', 'CERK5']
tolerances = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-8, 1e-10, 1e-12]


for Tol in tolerances:
    print('===========================================================')
    print(f'Tol = {Tol} \n')
    for method in methods:
        solution = solve_dde(t_span, f, alpha, phi, method = method, Atol=Tol, Rtol=Tol)

        max_diff = 0
        for i in range(len(solution.t) - 1):
            tt = np.linspace(solution.t[i], solution.t[i + 1], 100)
            sol = np.array([solution.eta(i) for i in tt])
            realsol = np.array([real_sol(i) for i in tt])
            max_diff = np.max(np.abs(realsol - sol))
            if max_diff > max_diff:
                max_diff = max_diff
        
        print(f'method = {method}')
        print('max diff', max_diff)
        print('steps: ', solution.steps)
        print('fails: ', solution.fails)
        print('feval: ', solution.feval)
        print('discs', solution.discs)
        print('')
