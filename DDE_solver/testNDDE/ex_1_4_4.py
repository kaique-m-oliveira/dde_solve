# from DDE_solver.rkh_state import *
# from DDE_solver.rkh_step_rejection import *
# from DDE_solver.rkh_testing import *
# from DDE_solver.rkh import *
# from DDE_solver.rkh_overlapping import *
# from DDE_solver.rkh_ovl_simp_newton import *
from DDE_solver.rkh_vectorize import *
# from DDE_solver.solve_dde import *
# from DDE_solver.rkh_NDDE import *

# WARN: STATE EXAMPLE


def f(t, y, yq):
    y1, y2, y3, y4, y5 = y
    yq1, yq2, yq3, yq4, yq5 = yq
    x1 = yq5 + yq3
    return np.array([x1, x2])


def phi(t):
    return np.array([np.exp(t), 1 - 1/np.e])


def alpha(t, y):
    return t - 1


def real_sol(t):
    return np.array([np.exp(t), np.exp(t) - np.exp(t - 1)])


def phi_t(t): return [0, 0]


t_span = [0, 3]


solver = Solver(f, alpha, phi, t_span)
solver.etas_t.append(phi_t)
solver.solve_dde()  # real_sol=real_sol)
tt = np.linspace(t_span[0], t_span[1], 100)
realsol = np.array([real_sol(t) for t in tt])
sol = np.array([solver.eta(i) for i in tt])
# for i in range(len(tt)):
#     print(tt[i], realsol[i] - sol[i])
print("max", np.max(np.abs(sol - realsol)))
solution = np.array([real_sol(t) for t in solver.t])


plt.plot(tt, realsol, color="red", label="realsol")
plt.plot(tt, sol, color="blue", label="aproxx")
plt.legend()
plt.show()
