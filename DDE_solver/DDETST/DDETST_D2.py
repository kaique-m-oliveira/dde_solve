import numpy as np

from DDE_solver.rkh_refactor import *

# Parameters
r1 = 0.02
r2 = 0.005
alpha_param = 3.0
delta = 0.01


def f(t, y, x):
    """
    Standard convention for a single delay:
    - x is the full delayed state vector evaluated at alpha(t,y) = t - y4(t)
    """
    y1, y2, y3, y4 = y
    xd1, xd2, xd3, xd4 = x  # delayed state: y(t - y4(t))

    dy1 = -r1 * y1 * y2 + r2 * y3
    dy2 = -r1 * y1 * y2 + alpha_param * r1 * xd1 * xd2
    dy3 = r1 * y1 * y2 - r2 * y3

    denom = xd1 * xd2 + xd3
    # Protect against division by zero (solver should handle, but keep safe)
    if denom == 0.0:
        dy4 = 1.0 + 0.0  # fallback; you may prefer np.inf or raise
    else:
        dy4 = 1.0 + ((3.0 * delta - y1 * y2 - y3) / denom) * np.exp(delta * y4)

    return [dy1, dy2, dy3, dy4]


def phi(t):
    # history for t <= 0
    return [5.0, 0.1, 0.0, 0.0]


def alpha(t, y):
    # single state-dependent delay: t - y4(t)
    y1, y2, y3, y4 = y
    return t - y4



t_span = [0.0, 40.0]


print(f'{'='*80}')
print(f''' {'='*80} 
      This is problem 1.1.10 from Paul
      ''')

methods = ['RKC3', 'RKC4', 'RKC5']
tolerances = [1e-2, 1e-4, 1e-6, 1e-8, 1e-10]


for Tol in tolerances:
    print('===========================================================')
    print(f'Tol = {Tol} \n')
    for method in methods:
        solution = solve_dde(f, alpha, phi, t_span, method = method, Atol=Tol, Rtol=Tol)
        
        print(f'method = {method}')
        print('No analytical solution')
        print('steps: ', solution.steps)
        print('fails: ', solution.fails)
        print('feval: ', solution.feval)
        print('')
