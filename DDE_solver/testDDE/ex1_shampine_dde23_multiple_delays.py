# from DDE_solver.rkh_state import *
# from DDE_solver.rkh_step_rejection import *
# from DDE_solver.rkh_testing import *
# from DDE_solver.rkh import *
import numpy as np
import matplotlib.pyplot as plt
# from DDE_solver.rkh_ovl_simp_newton import *
# from DDE_solver.rkh_fast_ov_test_disc import *
# from DDE_solver.rkh_vectorize import *
from DDE_solver.rkh_refactor import *
# from DDE_solver.rkh_state_complete import *

# from DDE_solver.rkh_NDDE import *

# WARN: STATE EXAMPLE


def f(t, y, delays):
    lag1, lag2 = delays
    x1 = -y[0] * lag1[1] - lag2[1]
    x2 = y[0] * lag1[1] - y[1]
    x3 = y[1] - lag2[1]
    return x1, x2, x3


def phi(t):
    return [5, 0.1, 1]


def alpha(t, y):
    return [t-1, t-10]


t_span = [0, 2]

d_f = [0, lambda t, y, x: [0, 0, 0], lambda t, y, x: [0, 0, 0]]


def alpha_t(t, y):
    return [1, 1]


def alpha_y(t, y):
    return [0, 0]


d_alpha = [alpha_t, alpha_y]
def d_phi(t): return 0


# tt = np.linspace(t_span[0], t_span[1], 100)
# realsol = np.array([real_sol(t) for t in tt])
# plt.plot(tt, realsol, color="red", label='real solution')
# plt.show()


solver = solve_dde(f, alpha, phi, t_span, d_f, d_alpha, d_phi)
tt = np.linspace(t_span[0], t_span[1], 100)
# realsol = np.array([real_sol(t) for t in tt])
sol = np.array([solver.eta(i) for i in tt])
# for i in range(len(tt)):
#     print(tt[i], realsol[i] - sol[i])
# print("max", np.max(abs(sol - realsol)))
# solution = np.array([real_sol(t) for t in solver.t])
# print('adnaed', np.max(np.squeeze(solver.y) - np.squeeze(solution)))


# plt.plot(tt, realsol, color="red", label='real solution')
plt.plot(tt, sol, color="blue", label='aproxx')
plt.legend()
plt.show()
