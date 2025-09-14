# from DDE_solver.rkh_state import *
# from DDE_solver.rkh_step_rejection import *
# from DDE_solver.rkh_testing import *
# from DDE_solver.rkh import *
import numpy as np
# from DDE_solver.rkh_ovl_simp_newton import *
# from DDE_solver.rkh_vectorize import *
# from DDE_solver.rkh_multiple_delays import *
from DDE_solver.rkh_refactor import *
# from DDE_solver.rkh_refactor_working import *
# from DDE_solver.rkh_state_complete import *
# from DDE_solver.rkh_fast_ov_test_disc import *
# from DDE_solver.rkh_NDDE import *

# WARN: STATE EXAMPLE


def f(t, y, yq):
    return 1 - yq


def phi(t):
    return np.log(t)


def alpha(t, y):
    return np.exp(1 - (1/t))


def real_sol(t):
    return np.log(t)


t_span = [0.5, 5]
def f_t(t): return 0
def f_y(t, y, x): return 0
def f_x(t, y, x): return -1


d_f = [f_t, f_y, f_x]
def alpha_t(t, y): return (np.e**(1 - 1 / t)) / t**2
def alpha_y(t, y): return 0


d_alpha = [alpha_t, alpha_y]
def d_phi(t): return 1 / t


print(f'{'='*80}')
print('ex_1_2_7.py')
solver = solve_dde(f, alpha, phi, t_span, d_f=d_f,
                   d_alpha=d_alpha, d_phi=d_phi)
# solver = Solver(f, alpha, phi, t_span)
# solver.f_y = f_y
# solver.f_x = f_x
# solver.alpha_y = alpha_y
# solver.phi_t = d_phi


tt = np.linspace(t_span[0], t_span[1], 100)
realsol = np.array([real_sol(t) for t in tt])
sol = np.array([solver.eta(i) for i in tt])
# for i in range(len(tt)):
#     print(tt[i], realsol[i] - sol[i])
sol_processed = np.squeeze(sol)
max_error = np.max(np.abs(sol_processed - realsol))
y = np.squeeze(solver.y)
solution = np.array([np.squeeze(real_sol(t)) for t in solver.t])
# input(f'realsol {realsol} type {type(realsol)} shape {realsol.shape}')
# input(f'sol {sol} type {type(sol)} shape {sol.shape}')
# input(f'solver.y {solver.y} type {type(solver.y)}')
# print('solution', solution)
# print('shape solver.y', solver.y)
#
# print(f"diferença em pontos aleatórios: {max_error}")
# print('diferença na malha:', np.max(y - solution))


plt.plot(tt, realsol, color="red", label='real solution')
plt.plot(tt, sol, color="blue", label='aproxx')
plt.legend()
plt.show()
