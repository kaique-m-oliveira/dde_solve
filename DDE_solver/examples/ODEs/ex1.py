# from DDE_solver.rkh_state import *
# from DDE_solver.rkh_step_rejection import *
# from DDE_solver.rkh_testing import *
# from DDE_solver.rkh import *
from DDE_solver.rkh_overlapping import *

# WARN: STATE EXAMPLE


def f(t, y, yq):
    return -y


def phi(t):
    return 1
    # return np.exp(-1)


def alpha(t, y):
    return -1e10


def real_sol(t):
    return np.exp(-t)


# t_span = [1, 2]
t_span = [0, 2]

solver = Solver(f, alpha, phi, t_span)
solver.solve_dde(real_sol=real_sol)
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
