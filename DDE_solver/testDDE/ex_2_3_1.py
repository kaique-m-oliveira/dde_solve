# from DDE_solver.rkh_state import *
# from DDE_solver.rkh_step_rejection import *
# from DDE_solver.rkh_testing import *
# from DDE_solver.rkh import *
import numpy as np
# from DDE_solver.rkh_ovl_simp_newton import *
# from DDE_solver.rkh_NDDE import *
from DDE_solver.rkh_vectorize import *

# WARN: STATE EXAMPLE


def f(t, y, yq, yz):
    return -yz


def phi(t):
    return 1 - t


def alpha(t, y):
    return t - (y**2)/2


def real_sol(t):
    return t + 1


t_span = [0, 2]

solver = Solver(f, alpha, phi, t_span)
solver.f_y = 0
solver.f_x = -1
solver.alpha_t = lambda t: 1
solver.alpha_y = lambda y: -y
solver.etas_t.append(lambda t: -1)


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
