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
    yq1, yq2 = yq
    return 1 - yq1 - yq2


def phi(t):
    return np.log(t)


def alpha(t, y):
    x1 = np.exp(1 - (1/t))
    x2 = t/np.e
    # print('x1', x1)
    # print('x2', x2)
    return [x1, x2]


def real_sol(t):
    return np.log(t)


t_span = [0.5, 5]


print(f'{'='*80}')
print('ex_1_2_7.py')
solver = solve_dde(f, alpha, phi, t_span)
# solver = Solver(f, alpha, phi, t_span)
# solver.f_y = f_y
# solver.f_x = f_x
# solver.alpha_y = alpha_y
# solver.phi_t = d_phi


tt = np.linspace(t_span[0], t_span[1], 100)
# realsol = np.array([real_sol(t) for t in tt])
sol = np.array([solver.eta(i) for i in tt])
# for i in range(len(tt)):
#     print(tt[i], realsol[i] - sol[i])
sol_processed = np.squeeze(sol)
# max_error = np.max(np.abs(sol_processed - realsol))
y = np.squeeze(solver.y)
# solution = np.array([np.squeeze(real_sol(t)) for t in solver.t])
# input(f'realsol {realsol} type {type(realsol)} shape {realsol.shape}')
# input(f'sol {sol} type {type(sol)} shape {sol.shape}')
# input(f'solver.y {solver.y} type {type(solver.y)}')
# print('solution', solution)
# print('shape solver.y', solver.y)
#
# print(f"diferença em pontos aleatórios: {max_error}")
# print('diferença na malha:', np.max(y - solution))


# plt.plot(tt, realsol, color="red", label='real solution')
plt.plot(tt, sol, color="blue", label='aproxx')
plt.legend()
plt.show()
