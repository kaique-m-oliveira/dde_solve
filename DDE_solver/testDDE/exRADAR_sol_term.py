import numpy as np
from DDE_solver.rkh_refactor import *


def f(t, y, yq):
    return -yq + 5


def phi(t):
    return 9/2 if t < -1 else -1/2


def alpha(t, y):
    return t - 2 - y**2


t_span = [0, 3]
discs = [(-1, 9/2, -1/2)]

solver = solve_dde(f, alpha, phi, t_span, discs=discs)
plt.plot(solver.t, solver.y, color="blue", label='aproxx')
plt.legend()
plt.show()
