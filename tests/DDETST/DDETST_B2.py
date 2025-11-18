import numpy as np

from dde_solve import *


def f(t, y, x):
    fx = 1.0 if x < 0 else -1.0
    return fx - y


def phi(t):
    return 1.0


def alpha(t, y):
    return t / 2.0


def real_sol(t):
    if 0 <= t <= 2*np.log(2):
        return 2*np.exp(-t) - 1
    elif 2*np.log(2) < t <= 2*np.log(6):
        return 1 - 6*np.exp(-t)
    elif 2*np.log(6) < t <= 2*np.log(66):
        return 66*np.exp(-t) - 1


t_span = [0, 2*np.log(66)]


methods = ['RKC3', 'RKC4', 'RKC5']
tolerances = [1e-3, 1e-4, 1e-5, 1e-6, 1e-8, 1e-10]


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
        print('discs: ', solution.discs)
        print('')
