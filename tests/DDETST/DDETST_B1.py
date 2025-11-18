import numpy as np

from dde_solve import *


def f(t, y, x):
    return 1 - x

def phi(t):
    return np.log(t)

def alpha(t, y):
    return np.exp(1 - 1/t)

def real_sol(t):
    return np.log(t)

t_span = [0.1, 10]

methods = ['CERK3', 'CERK4','CERK5']
tolerances = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-8, 1e-10, 1e-12]
# methods = ['RKC4', 'RKC5']
# tolerances = [1e-4, 1e-6, 1e-8, 1e-10, 1e-12]


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
            max_diff_ = np.max(np.abs(realsol - sol))
            if max_diff_ > max_diff:
                max_diff = max_diff_
        
        print(f'method = {method}')
        print('max diff', max_diff)
        print('steps: ', solution.steps)
        print('fails: ', solution.fails)
        print('feval: ', solution.feval)
        print('')
