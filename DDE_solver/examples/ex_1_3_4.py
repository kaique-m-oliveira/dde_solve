# from DDE_solver.rkh_state import *
from DDE_solver.rkh_step_rejection import *

# WARN: STATE EXAMPLE


def f(t, y, yq):
    return y*yq/t


def phi(t):
    return 1


def alpha(t, y):
    return np.log(y)


def real_sol(t):
    return t if 1 <= t <= np.exp(1) else np.exp(t / np.exp(1))


t_span = [1, np.exp(2)]
discs = [np.exp(1)]

solver = Solver(f, alpha, phi, t_span)
solver.solve_dde(discs=discs)

tt = np.linspace(t_span[0], t_span[1], 100)
realsol = np.array([real_sol(t) for t in tt])
sol = np.array([solver.eta(i) for i in tt])
print("max", max(abs(sol - realsol)))
#

plt.plot(tt, realsol, color="red")
plt.plot(tt, sol, color="blue")
plt.show()
