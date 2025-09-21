# from DDE_solver.rkh_state import *
# from DDE_solver.rkh_step_rejection import *
# from DDE_solver.rkh_testing import *
# from DDE_solver.rkh import *
# from DDE_solver.rkh_overlapping import *
# from DDE_solver.rkh_ovl_simp_newton import *
# from DDE_solver.rkh_vectorize import *
# from DDE_solver.rkh_multiple_delays import *
from DDE_solver.rkh_refactor import *
# from DDE_solver.solve_dde import *
# from DDE_solver.rkh_NDDE import *

# WARN: STATE EXAMPLE


def f(t, y, yq):
    y1, y2 = y
    yq1, yq2 = yq
    x1 = yq1[0] + y2
    x2 = y1 - yq2[1]
    # return np.array([x1, x2])
    return np.array([x1, x2])


def phi(t):
    # return np.array([np.exp(t), 1 - 1/np.e])
    return [np.exp(t), 1 - 1/np.e]


def alpha(t, y):
    return [np.exp(1 - (1/t)), t/np.e]


# def real_sol(t):
#     return np.array([np.exp(t), np.exp(t) - np.exp(t - 1)])


def phi_t(t): return [0, 0]


t_span = [0.5, 3]


print(f'{'='*80}')
print('ex_1_4_1_system.py')
solver = solve_dde(f, alpha, phi, t_span)
# solver = Solver(f, alpha, phi, t_span)
# solver.etas_t.append(phi_t)
# solver.solve_dde()  # real_sol=real_sol)
tt = np.linspace(t_span[0], t_span[1], 100)
# realsol = np.array([real_sol(t) for t in tt])
sol = np.array([solver.eta(i) for i in tt])
# for i in range(len(tt)):
#     print(tt[i], realsol[i] - sol[i])
# max_error = np.max(np.abs(sol - realsol))
y = solver.y
# solution = np.array([real_sol(t) for t in solver.t])
# print('solution', solution)
# print('shape solver.y', solver.y)
# print(f"diferença em pontos aleatórios: {max_error}")
# print('diferença na malha:', np.max(y - solution))


# plt.plot(tt, realsol, color="red", label="realsol")
plt.plot(tt, sol, color="blue", label="aproxx")
plt.legend()
plt.show()
