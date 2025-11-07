import numpy as np

from DDE_solver.rkh_refactor import *

L3 = 0.5
def f(t, y, x, z):
    val = np.cos(t)*(1 + x)+L3*y*z + (1 - L3)*np.sin(t)*np.cos(t*np.sin(t)**2) - np.sin(t + t*np.sin(t)**2)
    return val


def phi(t):
    return 0


def phi_t(t):
    return 1


def alpha(t, y):
    return t*(y**2)

beta = alpha


def real_sol(t):
    return np.sin(t)

t_span = [0, np.pi]

print(f'{'='*80}')
print(f''' {'='*80} 
      This is problem DDETST H2 
      ''')

methods = ['RKC3', 'RKC4', 'RKC5']
tolerances = [1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12]
# methods = ['RKC5']
# tolerances = [1e-4, 1e-6, 1e-8, 1e-10, 1e-12]


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
