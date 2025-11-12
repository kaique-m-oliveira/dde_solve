import numpy as np

from DDE_solver.rkh_refactor import *


def f(t, y, x):
    y1, y2 = y
    x1, x2 = x
    dy1 = y2
    dy2 = -x2*(y2**2)*np.exp(1 - y2)
    return [dy1, dy2]

def phi(t):
    return [np.log(t), 1/t]

def alpha(t, y):
    y1, y2 = y
    return np.exp(1 - y2)

def real_sol(t):
    return [np.log(t), 1/t]


t_span = [0.1, 5]


print(f'{'='*80}')
print(f''' {'='*80} 
      This is problem 1.1.10 from Paul
      ''')

methods = ['RKC3', 'RKC4', 'RKC5']
tolerances = [1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12]


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
        print('')
