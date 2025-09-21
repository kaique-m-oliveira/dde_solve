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
    results = []
    for ti in t:
        idx = bisect_left(ts, ti)
        if idx == 0:
            results.append(phi(ti))
        else:
            # Nesse caso, temos que t0 <= t <= t1
            t0 = ts[idx - 1]
            t1 = ts[idx]
            y0 = ys[idx - 1]
            y1 = ys[idx]
            theta = ti - t0
            results.append((1 - theta)*y0 + theta*y1)
    return results


def euler(t_span, f, alpha, phi, n):
    t0, tf = t_span
    t = np.linspace(t0, tf, n)
    h = (tf - t0)/n
    y = np.zeros(n)
    y[0] = phi(t[0])

    for i in range(n - 1):
        alpha_i = alpha(t[i], 0)
        Y1 = eta(alpha_i[0], t, y, phi)
        Y2 = eta(alpha_i[1], t, y, phi)
        Y3 = eta(alpha_i[2], t, y, phi)

        y[i + 1] = y[i] +  h * f(t[i], y[i], [Y1, Y2, Y3])
    return t, y

def mid_point(t_span, f, alpha, phi, n):
    t0, tf = t_span
    t = np.linspace(t0, tf, n)
    h = (tf - t0)/n
    y = np.zeros(n)
    y[0] = phi(t[0])

    for i in range(n - 1):
        eta_0 = eta(alpha(t[i], y[i]), t, y, phi)
        y_tilde = y[i] + h*f(t[i], y[i], eta_0)
        eta_1 = eta(alpha(t[i] + h, y_tilde), t, y, phi)
        first = f(t[i], y[i], eta_0)
        second = f(t[i] + h, y_tilde, eta_1)
        y[i + 1] = y[i] + 0.5 * h * ( first + second)
    return t, y


#
# t_span = [0, 8]
# def f(t, y, x): return -x[0] - x[1] - x[2]
# def alpha(t, y): return [t - 1, t-2, t-3]
# def phi(t): return 1
#
#
# def real_sol(t):
#     if 0 <= t <= 1:
#         return 1 - t
#     if 1 <= t <= 2:
#         return (1/2)*(t**2 - 4*t + 3)
#     if 2 <= t <= 3:
#         return (1/6) * (17 - 24*t + 9*t**2 - t**3)
#     return 0

def f(t, y, x):
    x1, x2, x3, x4 = x
    return -x1 + x2 - x3*x4


def phi(t):
    if t < 0:
        return 1
    else:
        return 0


def alpha(t, y):
    return [t-1,  t-2, t-3, t-4]


def real_sol(t):

    if 0 <= t <= 1:
        return -t
    elif 1 < t <= 2:
        return (1/2) * t**2 - t - (1/2)
    elif 2 < t <= 3:
        return (-1/6) * t**3 + (1/2) * t**2 - (7/6)
    elif 3 < t <= 4:
        return (1/24) * t**4 - (1/6) * t**3 - (1/4) * t**2 + t - (19/24)
    elif 4 < t <= 5:
        return (-1/120) * t**5 + (1/6) * t**4 - (5/3) * t**3 + (109/12) * t**2 - 24 * t + (2689/120)
    else:
        return np.nan


t_span = [0, 5]


ts0, ys0 = mid_point(t_span, f, alpha, phi, 250)
ts1, ys1 = mid_point(t_span, f, alpha, phi, 300)
print(f'ts0 {ts0}')
input(f'ts1 {ts1}')

solver = solve_dde(f, alpha, phi, t_span)
my_approx0 = np.array([solver.eta(t) for t in ts0])
my_approx1 = np.array([solver.eta(t) for t in ts1])

max_diff0 = np.max(np.abs(my_approx0 - ys0))
max_diff1 = np.max(np.abs(my_approx1 - ys1))
print('max_diff0', max_diff0)
print('max_diff1', max_diff1)

realsolution = [real_sol(t) for t in ts1]


plt.plot(ts0, my_approx0, color="red", label='my method')
plt.plot(ts1, realsolution, color="orange", label='real solution')
plt.plot(ts0, ys0, color="green", label='ys0')
plt.plot(ts1, ys1, color="blue", label='ys1')
plt.legend()
plt.show()
