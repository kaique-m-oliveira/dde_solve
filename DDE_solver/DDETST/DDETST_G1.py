import numpy as np

from DDE_solver.rkh_refactor import *


# def f(t, y, x, z):
#     return -4*t*y**2 / 4 + np.log(np.cos(2*t))**2 + np.tan(2*t) + 0.5*np.arctan(z)
def f(t, y, x, z):
    return -z


def phi(t):
    return 1 - t


def phi_t(t):
    return -1


def alpha(t, y):
    return t - (1/4)*y**2


beta = alpha


def real_sol(t):
    return t + 1

t_span = [0, 1]

print(f'{'='*80}')
print(f''' {'='*80} 
      This is problem 1.3.4 from Paul
      ''')

methods = ['RKC3', 'RKC4', 'RKC5']
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
            max_diff = np.max(np.abs(realsol - sol))
            if max_diff > max_diff:
                max_diff = max_diff
        
        print(f'method = {method}')
        print('max diff', max_diff)
        print('steps: ', solution.steps)
        print('fails: ', solution.fails)
        print('feval: ', solution.feval)
        print('')

