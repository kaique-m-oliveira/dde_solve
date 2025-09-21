# from DDE_solver.rkh_state import *
# from DDE_solver.rkh_step_rejection import *
# from DDE_solver.rkh_testing import *
# from DDE_solver.rkh import *
import numpy as np
from DDE_solver.rkh_ovl_simp_newton import *

# WARN: STATE EXAMPLE


def f(t, y, yq):
    return -y + yq + (1/20)*np.cos(t/20) + np.sin(t/20) - np.sin(t - 1 + np.sin(t))


def phi(t):
    return np.sin(t/20)


def alpha(t, y):
    return t - 1 + np.sin(t)


def real_sol(t):
    return np.sin(t/20)


t_span = [0, 1]

solver = Solver(f, alpha, phi, t_span)
# solver.f_y = -1
# solver.f_x = 1
# solver.alpha_t = lambda t: 1 + np.cos(t)
# solver.alpha_y = 0
# solver.phi_t = lambda t: (t/20)*np.cos(t/20)
# solver.etas_t.append(lambda t: (t/20)*np.cos(t/20))


solver.solve_dde()
tt = np.linspace(t_span[0], t_span[1], 100)
realsol = np.array([real_sol(t) for t in tt])
sol = np.array([solver.eta(i) for i in tt])
# for i in range(len(tt)):
#     print(tt[i], realsol[i] - sol[i])
print("max", max(abs(sol - realsol)))
solution = np.array([real_sol(t) for t in solver.t])
print('adnaed', max(solver.y - solution))


plt.plot(tt, realsol, color="red", label='real solution')
plt.plot(tt, sol, color="blue", label='aproxx')
plt.legend()
plt.show()
