import numpy as np

from DDE_solver.rkh_refactor import *

# Parameters
s0_hat = 0.00372
T1 = 3
gamma = 0.1
Q = 0.00178
k = 6.65
a = 15600
K = 0.0382
r = 6.96


def f_nl(y1):
    """Nonlinear production function f(y1) = a / (1 + K y1^r)."""
    return a / (1 + K * y1**r)


def f(t, y, x):
    """
    Right-hand side of the DDE system.
    x corresponds to the delayed state [y1(t - τ1), y2(t - τ2), y3(t - τ3)] as needed.
    """
    y1, y2, y3 = y
    x1, x2 = x
    x11, x12, x13 = x1
    x21, x22, x23 = x2

    dy1 = s0_hat * x12 - gamma * y1 - Q
    dy2 = f_nl(y1) - k * y2
    dy3 = 1 - (Q * np.exp(gamma * y3)) / (s0_hat * x22)
    return [dy1, dy2, dy3]


def phi(t):
    return [3.5, 10, 50]


def alpha(t, y):
    """Delay functions for each equation."""
    y1, y2, y3 = y
    return [t - T1, t - T1 - y3]



t_span = [0, 100]


print(f'{'='*80}')
print(f''' {'='*80} 
      This is problem 1.2.6 from Paul
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
