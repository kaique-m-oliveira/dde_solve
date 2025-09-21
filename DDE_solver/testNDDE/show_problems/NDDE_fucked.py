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


def eta_rk(t, ts, ys, K1s, K5s, phi):
    idx = bisect_left(ts, t)
    if idx == 0:
        return phi(t)
    t0, t1 = ts[idx - 1], ts[idx]
    y0, y1 = ys[idx - 1], ys[idx]
    k1, k5 = K1s[idx], K5s[idx]

    h = t1 - t0
    theta = (t - t0)/h

    d1 = (theta - 1)**2 * (2*theta + 1)
    d2 = theta**2 * (3 - 2*theta)
    d3 = theta * (theta - 1)**2
    d4 = theta**2 *(theta - 1)

    return d1*y0 + d2*y1 + d3*h*k1 + d4*h*k5


def eta_t_rk(t, ts, ys, K1s, K5s, phi, phi_t):
    idx = bisect_left(ts, t)
    if idx == 0:
        return phi_t(t)
    t0, t1 = ts[idx-1], ts[idx]
    y0, y1 = ys[idx-1], ys[idx]
    k1, k5 = K1s[idx], K5s[idx]

    h = t1 - t0
    theta = (t - t0)/h

    d1 = 6*theta**2 - 6*theta
    d2 = 6*theta - 6*theta**2
    d3 = 3*theta**2 - 4*theta + 1
    d4 = 3*theta**2 - 2*theta

    return (d1*y0 + d2*y1)/h + d3*k1 + d4*k5


def rk4(t_span, f, alpha, phi, phi_t, n):
    t0, tf = t_span
    t = np.linspace(t0, tf, n)
    h = (tf - t0)/n
    y = np.zeros(n)
    K1s = np.zeros(n)
    K5s = np.zeros(n)
    y[0] = phi(t[0])
    for i in range(n-1):
        print('[t0, t1] = ', t[i], t[i+1])
        Y_tilde = eta_rk(alpha(t[i], y[i]), t, y,  K1s, K5s, phi)
        Z_tilde = eta_t_rk(alpha(t[i], y[i]), t, y,  K1s, K5s, phi, phi_t)
        # Z_tilde = eta_t_mid(alpha(t[i], y[i]), t, y, phi, phi_t)
        k1 = f(t[i], y[i], Y_tilde, Z_tilde)
        K1s[i + 1] = k1

        t2, y2 = t[i] + 0.5*h, y[i] + 0.5*h*k1
        Y_tilde = eta_rk(alpha(t2, y2), t, y,  K1s, K5s, phi)
        Z_tilde = eta_t_rk(alpha(t2, y2), t, y,  K1s, K5s, phi, phi_t)
        # Z_tilde = eta_t_mid(alpha(t2, y2), t, y, phi, phi_t)
        real_t = real_sol_t(alpha(t2, y2))
        diff = abs(Z_tilde - real_t)
        # input(f"Z_tilde, {Z_tilde}, real, {real_t}, diff {diff}")
        k2 = f(t2, y2, Y_tilde, Z_tilde)

        t3, y3 = t[i] + 0.5*h, y[i] + 0.5*h*k2
        Y_tilde = eta_rk(alpha(t3, y3), t, y,  K1s, K5s, phi)
        Z_tilde = eta_t_rk(alpha(t3, y3), t, y,  K1s, K5s, phi, phi_t)
        # Z_tilde = eta_t_mid(alpha(t3, y3), t, y, phi, phi_t)
        k3 = f(t3, y3, Y_tilde, Z_tilde)

        t4, y4 = t[i] + h, y[i] + h*k3
        Y_tilde = eta_rk(alpha(t4, y4), t, y,  K1s, K5s, phi)
        Z_tilde = eta_t_rk(alpha(t4, y4), t, y,  K1s, K5s, phi, phi_t)
        # Z_tilde = eta_t_mid(alpha(t4, y4), t, y, phi, phi_t)
        k4 = f(t4, y4, Y_tilde, Z_tilde)

        y[i + 1] = y[i] + h*(k1 + 2 * k2 + 2 * k3 + k4) / 6
        input(f'y1 - real_sol {y[i + 1] - real_sol(t[i + 1])}')

        t5, y5 = t[i] + h, y[i + 1]
        Y_tilde = eta_rk(alpha(t5, y5), t, y,  K1s, K5s, phi)
        Z_tilde = eta_t_rk(alpha(t5, y5), t, y,  K1s, K5s, phi, phi_t)
        # Z_tilde = eta_t_mid(alpha(t5, y5), t, y, phi, phi_t)
        k5 = f(t5, y5, Y_tilde, Z_tilde)
        K5s[i + 1] = k5

    return t, y


#WARN: midpoint method with linear interpolation
def eta_mid(t, ts, ys, phi):
    idx = bisect_left(ts, t)
    if idx == 0:
        return phi(t)
    else:
        t0 = ts[idx - 1]
        t1 = ts[idx]
        y0 = ys[idx - 1]
        y1 = ys[idx]
        theta = (t - t0)/(t1 - t0)
        return (1 - theta)*y0 + theta*y1

def eta_t_mid(t, ts, ys, phi, phi_t):
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
        eta_0 = eta_mid(alpha(t[i], y[i]), t, y, phi)
        eta_0_t = eta_t_mid(alpha(t[i], y[i]), t, y, phi, phi_t)
        y_tilde = y[i] + h*f(t[i], y[i], eta_0, eta_0_t)
        eta_1 = eta_mid(alpha(t[i] + h, y_tilde), t, y, phi)
        eta_1_t = eta_t_mid(alpha(t[i] + h, y_tilde), t, y, phi, phi_t)
        first = f(t[i], y[i], eta_0, eta_0_t)
        second = f(t[i] + h, y_tilde, eta_1, eta_1_t)
        y[i + 1] = y[i] + 0.5 * h * (first + second)
    return t, y



def f(t, y, x, z):
    return y + x - 2*z

def phi(t): 
    if t<= 0:
        return -t
    elif 0 <= t <= 1:
        return -2 + t + 2*np.exp(t)

def phi_t(t):
    if t <= 0:
        return -1
    if 0 <= t <= 1:
        return 1 + 2*np.exp(t)

def alpha(t, y):
    return t-1

def real_sol(t):
    if t<= 0:
        return -t
    elif 0 <= t <= 1:
        return -2 + t + 2*np.exp(t)
    elif 1 <= t <= 2:
        return 4 - t + 2*np.exp(t) - 2*(t + 1)*np.exp(t - 1)


def real_sol_t(t):
    if t <= 0:
        return -1
    if 0 <= t <= 1:
        return 1 + 2*np.exp(t)
    if 1 <= t <= 2:
        return -1 + 2*np.exp(t) - 2*(t+2)*np.exp(t - 1)


epsilon = 0.01
t_span = [1 - epsilon, 2]


h = 0.03
n = int((t_span[-1] - t_span[0])/h)
ts, y_rk = rk4(t_span, f, alpha, phi, phi_t, n)
t_mid, y_mid = mid_point(t_span, f, alpha, phi, phi_t, n)

solver = solve_dde(f, alpha, phi, t_span, neutral = True, beta = alpha, d_phi = phi_t )
my_approx0 = np.array([solver.eta(t) for t in ts])
realsolution = [real_sol(t) for t in ts]


plt.plot(ts, my_approx0, color="red", label='my method')
plt.plot(ts, realsolution, color="orange", label='real solution')
plt.plot(ts, y_rk, color="green", label='RK')
plt.plot(t_mid, y_mid, color="blue", label='mid point')
plt.legend()
plt.show()



