# from DDE_solver.rkh_state import *
# from DDE_solver.rkh_step_rejection import *
# from DDE_solver.rkh_testing import *
# from DDE_solver.rkh import *
import numpy as np
from DDE_solver.rkh_ovl_simp_newton import *
# from DDE_solver.rkh_NDDE import *

# WARN: STATE EXAMPLE


def f(t, y, yq):
    return -yq


def phi(t):
    return 1


def alpha(t, y):
    return t - 1


def real_sol(t):
    if 0 <= t <= 1:
        return 1 - t
    if 1 <= t <= 2:
        return (1/2)*(t**2 - 4*t + 3)
    if 2 <= t <= 3:
        return (1/6) * (17 - 24*t + 9*t**2 - t**3)
    return 0


t_span = [0, 3]

solver = Solver(f, alpha, phi, t_span)
solver.f_y = lambda t, y, x: 0
solver.f_x = lambda t, y, x: -1
solver.alpha_t = lambda t, y: 1
solver.alpha_y = lambda t, y: 0
solver.phi_t = lambda t: 0
solver.etas_t.append(lambda t: 0)


solver.solve_dde()
tt = np.linspace(t_span[0], t_span[1], 100)
realsol = np.array([real_sol(t) for t in tt])
sol = np.array([solver.eta(i) for i in tt])
# for i in range(len(tt)):
#     print(tt[i], realsol[i] - sol[i])
solution = np.array([real_sol(t) for t in solver.t])
print('solution type', type(solution), len(solution), np.array(solution).shape)
print('solver.y type', type(solver.y), len(solver.y), solver.y)
print('sub', np.array(solver.y) - solution)
print("max", np.max(np.abs(sol - realsol)))
print('adnaed', np.max(np.abs(np.array(solver.y) - solution)))


plt.plot(tt, realsol, color="red", label='real solution')
plt.plot(tt, sol, color="blue", label='aproxx')
plt.legend()
plt.show()
