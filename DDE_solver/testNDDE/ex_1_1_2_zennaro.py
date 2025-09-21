# from DDE_solver.rkh_state import *
# from DDE_solver.rkh_step_rejection import *
# from DDE_solver.rkh_testing import *
# from DDE_solver.rkh import *
import numpy as np
# from DDE_solver.rkh_ovl_simp_newton import *
# from DDE_solver.rkh_fast_overlapping import *
# from DDE_solver.rkh_vectorize import *
from DDE_solver.rkh_refactor import *
# from DDE_solver.rkh_multiple_delays import *
# from DDE_solver.rkh_NDDE import *

# WARN: STATE EXAMPLE


def f(t, y, yq):
    return yq*(y - 1)


def phi(t):
    return 1


def alpha(t, y):
    return t - 1


def real_sol(t):
    return 1


t_span = [0, 3]

d_f = [0, lambda t, y, x: 0, lambda t, y, x: -1]
d_alpha = [lambda t, y: 1, lambda t, y: 0]
d_phi = [lambda t: 0]
solver = solve_dde(f, alpha, phi, t_span, d_f=d_f,
                   d_alpha=d_alpha, d_phi=d_phi)
# solver = Solver(f, alpha, phi, t_span, d_f=d_f, d_alpha=d_alpha, d_phi=d_phi)

print(f'{'='*80}')
print('ex_1_1_2_zennaro.py')
# solver.solve_dde()
tt = np.linspace(t_span[0], t_span[1], 100)
realsol = np.array([real_sol(t) for t in tt])
sol = np.array([solver.eta(i) for i in tt])
# for i in range(len(tt)):
#     print(tt[i], realsol[i] - sol[i])

print("max", np.max(np.abs(sol - realsol)))
solution = np.array([real_sol(t) for t in solver.t])
print('solution', solution)
print('shape solver.y', solver.y)

print('adnaed', np.max(solver.y - solution))


plt.plot(tt, realsol, color="red", label='real solution')
plt.plot(tt, sol, color="blue", label='aproxx')
plt.legend()
plt.show()
