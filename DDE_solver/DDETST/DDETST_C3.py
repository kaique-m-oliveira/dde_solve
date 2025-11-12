import numpy as np

from DDE_solver.rkh_refactor import *

# Parameters
s0_hat = 0.0031
T1 = 6.0
gamma = 0.001
Q = 0.0275
k = 2.8
a = 6570
K = 0.0382
r = 6.96


def f_nl(y1):
    return a / (1 + K * y1**r)


def f(t, y, x):
    y1, y2, y3 = y
    x1, x2 = x
    x11, x12, x13 = x1
    x21, x22, x23 = x2

    dy1 = s0_hat * x12 - gamma * y1 - Q
    dy2 = f_nl(y1) - k * y2
    dy3 = 1 - (Q * np.exp(gamma * y3)) / (s0_hat * x22)
    return [dy1, dy2, dy3]


def phi(t):
    if t <= -T1:
        phi2 = 9.5
    elif -T1 < t <= 0:
        phi2 = 10.0
    else:
        phi2 = np.nan  # outside history domain

    return [3.325, phi2, 120.0]


def alpha(t, y):
    """Delay functions for each equation."""
    y1, y2, y3 = y
    return [t - T1, t - T1 - y3]



t_span = [0, 300]


methods = ['RKC3', 'RKC4', 'RKC5']
tolerances = [1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12]
# methods = ['RKC4', 'RKC5']
# tolerances = [1e-12]


for Tol in tolerances:
    print('===========================================================')
    print(f'Tol = {Tol} \n')
    for method in methods:
        solution = solve_dde(t_span, f, alpha, phi, method = method, Atol=Tol, Rtol=Tol)

        
        print(f'method = {method}')
        print('No analytical solution')
        print('steps: ', solution.steps)
        print('fails: ', solution.fails)
        print('feval: ', solution.feval)
        print('')
