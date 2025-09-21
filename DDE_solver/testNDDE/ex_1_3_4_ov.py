# from DDE_solver.rkh_state import *
# from DDE_solver.rkh_step_rejection import *
# from DDE_solver.rkh_overlapping import *
# from DDE_solver.rkh_ovl_simp_newton import *
from DDE_solver.rkh_vectorize import *
# from DDE_solver.solve_dde import *

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
d_f = [lambda t, y, x: -(y * x)/t**2, lambda t, y, x: x/t, lambda t, y, x: y/t]
d_alpha = [lambda t, y: 0, lambda t, y: 1/y]
def d_phi(t): return 1


solver = Solver(f, alpha, phi, t_span, d_f, d_alpha, d_phi)
solver.solve_dde(discs=discs)

tt = np.linspace(t_span[0], t_span[1], 100)
realsol = np.array([real_sol(t) for t in tt])
sol = np.array([solver.eta(i) for i in tt])

sol_processed = np.squeeze(sol)
max_error = np.max(np.abs(sol_processed - realsol))
y = np.squeeze(solver.y)
solution = np.array([real_sol(t) for t in solver.t])
print(f"diferença em pontos aleatórios: {max_error}")
print('diferença na malha:', np.max(y - solution))
#

plt.plot(tt, realsol, color="red", label='real solution')
plt.plot(tt, sol, color="blue", label='aproxx')
plt.show()
