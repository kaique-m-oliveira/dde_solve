# from DDE_solver.rkh_state import *
# from DDE_solver.rkh_step_rejection import *
# from DDE_solver.rkh_testing import *
# from DDE_solver.rkh import *
from DDE_solver.rkh_ovl_simp_newton import *

# WARN: STATE EXAMPLE


def f(t, y, yq):
    return -y + yq*(2.5 - 1.5 * (yq/1000)**(2.5))


def phi(t):
    return 999


def alpha(t, y):
    return t - 2


def real_sol(t):
    return np.sin(t/20)


t_span = [0, 10]

solver = Solver(f, alpha, phi, t_span)
solver.solve_dde()
tt = np.linspace(t_span[0], t_span[1], 100)
realsol = np.array([real_sol(t) for t in tt])
sol = np.array([solver.eta(i) for i in tt])
# for i in range(len(tt)):
#     print(tt[i], realsol[i] - sol[i])
print("max", max(abs(sol - realsol)))
solution = np.array([real_sol(t) for t in solver.t])
print('adnaed', max(solver.y - solution))


plt.plot(tt, realsol, color="red")
plt.plot(tt, sol, color="blue")
plt.show()
