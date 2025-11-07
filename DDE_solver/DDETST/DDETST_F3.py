import numpy as np

from DDE_solver.rkh_refactor import *

# ---- Parameters ----
L3 = 0.2

# ---- Delay function ----
def alpha(t, y):
    return 0.5 * t * (1 - np.cos(2 * np.pi * t))

beta = alpha  # for neutral formulation (z uses same delay)

# ---- RHS definition ----
def f(t, y, x, z):
    return np.exp(-y) + L3 * (np.sin(z) - np.sin(1 / (3 + alpha(t, y))))

# ---- History functions ----
def phi(t):
    return np.log(3.0)

def phi_t(t):
    return 1/3.0

# ---- Analytical solution ----
def real_sol(t):
    return np.log(t + 3.0)

# ---- Problem setup ----
t_span = [0.0, 10.0]

print(f'{'='*80}')
print(f''' {'='*80} 
      This is problem 1.3.4 from Paul
      ''')

methods = ['RKC3', 'RKC4', 'RKC5']
tolerances = [1e-2,  1e-4, 1e-6, 1e-8, 1e-10]
# methods = ['RKC4', 'RKC5']
tolerances = [1e-2,  1e-4, 1e-6, 1e-8, 1e-10, 1e-12]

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

