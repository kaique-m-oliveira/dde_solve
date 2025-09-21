# isort: skip_file
# You can also use specific directives for other formatters
# like: # fmt: off

import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)
from rkh_refactor import solve_dde
import numpy as np
import matplotlib.pyplot as plt
from bisect import bisect_left


def eta(t, ts, ys, phi):
    idx = bisect_left(ts, t)
    if idx == 0:
        return phi(t)
    else:
        t0 = ts[idx - 1]
        t1 = ts[idx]
        y0 = ys[idx - 1]
        y1 = ys[idx]
        theta = t - t0
        return (1 - theta)*y0 + theta*y1

def eta_t(t, ts, ys, phi, phi_t):
    idx = bisect_left(ts, t)
    if idx == 0:
        return phi_t(t)
    else:
        t0 = ts[idx - 1]
        t1 = ts[idx]
        y0 = ys[idx - 1]
        y1 = ys[idx]
        return (y1 - y0)/(t1 - t0)


def mid_point(t_span, f, alpha, phi, phi_t, n):
    t0, tf = t_span
    t = np.linspace(t0, tf, n)
    h = (tf - t0)/n
    y = np.zeros(n)
    y[0] = phi(t[0])

    for i in range(n - 1):
        eta_0 = eta(alpha(t[i], y[i]), t, y, phi)
        eta_0_t = eta_t(alpha(t[i], y[i]), t, y, phi, phi_t)
        y_tilde = y[i] + h*f(t[i], y[i], eta_0, eta_0_t)
        eta_1 = eta(alpha(t[i] + h, y_tilde), t, y, phi)
        eta_1_t = eta_t(alpha(t[i] + h, y_tilde), t, y, phi, phi_t)
        first = f(t[i], y[i], eta_0, eta_0_t)
        second = f(t[i] + h, y_tilde, eta_1, eta_1_t)
        y[i + 1] = y[i] + 0.5 * h * ( first + second)
    return t, y


def f(t, y, x, z):
    return y + x - 2*z

def phi(t): 
    return -t

def phi_t(t):
    return -1

def alpha(t, y):
    return t-1

def real_sol(t):
    if 0 <= t <= 1:
        return -2 + t + 2*np.exp(t)
    elif 1 <= t <= 2:
        return 4 - t + 2*np.exp(t) - 2*(t + 1)*np.exp(t - 1)


t_span = [0, 2]


ts0, ys0 = mid_point(t_span, f, alpha, phi, phi_t, 250)
# ts1, ys1 = mid_point(t_span, f, alpha, phi, phi_t, 300)
# print(f'ts0 {ts0}')
# input(f'ts1 {ts1}')

solver = solve_dde(f, alpha, phi, t_span, neutral = True, beta = alpha, d_phi = phi_t )
my_approx0 = np.array([solver.eta(t) for t in ts0])
# my_approx1 = np.array([solver.eta(t) for t in ts1])

# max_diff1 = np.max(np.abs(my_approx1 - ys1))
# print('max_diff0', max_diff0)
# print('max_diff1', max_diff1)

realsolution = [real_sol(t) for t in ts0]


plt.plot(ts0, my_approx0, color="red", label='my method')
plt.plot(ts0, realsolution, color="orange", label='real solution')
plt.plot(ts0, ys0, color="green", label='ys0')
plt.legend()
plt.show()
