import numpy as np
import matplotlib.pyplot as plt
from DDE_solver.rkh_refactor import *


def f(t, y, x):
    x1, x2 = x
    return x1 + x2


def phi(t):
    if t < -1:
        return -1
    elif t == -1:
        return 0
    else:
        return 1


def alpha(t, y):
    return [t-1,  t-2]


def real_sol(t):
    if 0 <= t <= 1:
        return 1
    elif 1 <= t <= 2:
        return 2*t - 1
    elif 2 < t <= 3:
        return t**2 - 2*t + 3


t_span = [0, 3]

discs = [(-1, -1, 1)]

solver = solve_dde(f, alpha, phi, t_span, discs=discs)
tt = np.linspace(t_span[0], t_span[1], 100)
realsol = np.array([real_sol(t) for t in tt])
sol = np.array([solver.eta(i) for i in tt])
solution = np.array([real_sol(t) for t in solver.t])


plt.plot(tt, realsol, color="red", label='real solution')
plt.plot(tt, sol, color="blue", label='aproxx')
plt.legend()
plt.show()
