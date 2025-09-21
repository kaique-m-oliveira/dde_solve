# from DDE_solver.rkh_state import *
# from DDE_solver.rkh_step_rejection import *
# from DDE_solver.rkh_testing import *
# from DDE_solver.rkh import *
import numpy as np
# from DDE_solver.rkh_ovl_simp_newton import *
# from DDE_solver.rkh_fast_ov_test_disc import *
from DDE_solver.rkh_refactor import *
# from DDE_solver.rkh_NDDE import *

# WARN: STATE EXAMPLE


def f(t, y, yq):
    return 5*y + yq


def phi(t):
    return 5


def alpha(t, y):
    return t - 1


def real_sol(t):
    if 0 <= t <= 1:
        return 6*np.exp(5*t) - 1
    if 1 < t <= 2:
        return 6*(np.exp(5) + t - 6/5)*np.exp(5*t - 5) + 1/5


t_span = [0, 2]


def f_y(t, y, x): return 5
def f_x(t, y, x): return 1


d_f = [0, f_y, f_y]
def alpha_t(t, y): return 1
def alpha_y(t, y): return 0


d_alpha = [alpha_t, alpha_y]
def phi_t(t): return 0


solver = solve_dde(f, alpha, phi, t_span, d_f=d_f,
                   d_alpha=d_alpha, d_phi=phi_t)

tt = np.linspace(t_span[0], t_span[1], 100)
realsol = np.array([real_sol(t) for t in tt])
sol = np.array([solver.solution(i) for i in tt])
# for i in range(len(tt)):
#     print(tt[i], realsol[i] - sol[i])
print('realsol', realsol)
print('sol', sol)
print("max", max(abs(np.squeeze(sol) - realsol)))
solution = np.array([real_sol(t) for t in solver.t])
print('adnaed', max(np.squeeze(solver.y) - solution))


plt.plot(tt, realsol, color="red", label='real solution')
plt.plot(tt, sol, color="blue", label='aproxx')
plt.legend()
plt.show()
