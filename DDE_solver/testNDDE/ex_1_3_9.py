# from DDE_solver.rkh_state import *
# from DDE_solver.rkh_step_rejection import *
# from DDE_solver.rkh_ovl_simp_newton import *
from DDE_solver.rkh_vectorize import *

# WARN: STATE EXAMPLE


def f(t, y, yq):
    return -yq*y


def phi(t):
    return 1


def alpha(t, y):
    return t - np.log(1 + 2 * abs(y) * np.exp(1 - 2*np.exp(-t)))


def real_sol(t):
    if 0 <= t <= np.log(2):
        return np.exp(-t)
    else:
        return (1/2) * np.exp(2 * np.exp(-t) - 1)


t_span = [0, 1]
discs = [np.log(2)]

solver = Solver(f, alpha, phi, t_span)
solver.solve_dde(discs=discs)

tt = np.linspace(t_span[0], t_span[1], 100)
realsol = np.array([real_sol(t) for t in tt])
sol = np.array([solver.eta(i) for i in tt])
print("max", max(abs(sol - realsol)))
#

print('initial value t = 0 for real_sol and solver', real_sol(0), solver.eta(0))


plt.plot(tt, realsol, color="red")
plt.plot(tt, sol, color="blue")
plt.show()
