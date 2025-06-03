import matplotlib.pyplot as plt
import numpy as np
import scipy as sc


def rk4(f, t, y0):
    N = len(t)
    y = np.zeros(N)
    t[0], y[0] = t[0], y0

    for n in range(N - 1):
        h = t[n + 1] - t[n]
        k1 = h * f(t[n], y[n])
        k2 = h * f(t[n] + 0.5 * h, y[n] + 0.5 * k1)
        k3 = h * f(t[n] + 0.5 * h, y[n] + 0.5 * k2)
        k4 = h * f(t[n] + h, y[n] + k3)
        y[n + 1] = y[n] + (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return y


def interpolation(x, y, x0):
    n = len(x)
    y0 = 0
    for i in range(n):
        L = 1
        for j in range(n):
            if j != i:
                L *= (x0 - x[j]) / (x[i] - x[j])
        y0 += y[i] * L
    return y0


def f1(t, y):
    return y - t**2 + 1


def f2(t, y):
    return y * np.cos(t)


y0 = 0.5
t_0, t_f = 0, 4
t = np.arange(0, t_f, 0.01)
solution = sc.integrate.odeint(f1, y0, t)
sol = sc.integrate.solve_ivp(f1, [0, t_f], [y0], t_eval=t)

y = rk4(f1, t, y0)

plt.plot(t, y)
# plt.plot(t, solution)
# plt.plot(t, sol.y[0])
plt.show()
