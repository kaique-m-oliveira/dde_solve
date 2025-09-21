# from DDE_solver.rkh_state import *
from DDE_solver.rkh_step_rejection import *

# WARN: STATE EXAMPLE


def f(t, y, yq):
    return yq


def phi(t):
    return t**2


def alpha(t, y):
    return t - y(t - t**2)


def fun(t):
    return (1/9)*t**9 - (1/2)*t**8 + (6/7)*t**(7) - t**(6) + t**(5) - (1/2)*t**(4) + (1/3)*t**(3)


def real_sol(t, gamma=1.75487766624669):
    return t if 1 <= t <= gamma else fun(t) - fun(gamma)


gamma = 1.75487766624669
t_span = [1, 2*gamma]

solver = Solver(f, alpha, phi, t_span)
solver.solve_dde()

tt = np.linspace(t_span[0], t_span[1], 100)
realsol = np.array([real_sol(t) for t in tt])
sol = np.array([solver.eta(i) for i in tt])
print("max", max(abs(sol - realsol)))
#

plt.plot(tt, realsol, color="red")
plt.plot(tt, sol, color="blue")
plt.show()
