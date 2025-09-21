# from DDE_solver.rkh_state import *
# from DDE_solver.rkh_step_rejection import *
# from DDE_solver.rkh_overlapping import *
# from DDE_solver.rkh_ovl_simp_newton import *
from DDE_solver.rkh_vectorize import *

# WARN: STATE EXAMPLE


def f(t, y, yq):
    return yq/(2 * np.sqrt(t))


def phi(t):
    return 1


def alpha(t, y):
    return y - np.sqrt(2) + 1


def real_sol(t):
    if 1 <= t <= 2:
        return np.sqrt(t)
    else:
        return (1/4)*t + (1/2) + (1 - 1/np.sqrt(2))*np.sqrt(t)


discs = [2, 5.0294372515148]
t_span = [1, discs[-1]]

solver = Solver(f, alpha, phi, t_span)
solver.solve_dde(discs=discs)

tt = np.linspace(t_span[0], t_span[1], 10)
realsol = np.array([real_sol(t) for t in tt])
sol = np.array([solver.eta(i) for i in tt])
for i in range(len(tt)):
    print(tt[i], realsol[i] - sol[i])
#
# print('t values', solver.t)
print('diferença no t = 1.999', abs(solver.eta(1.999) - real_sol(1.999)))
print('diferença no t = 2', abs(solver.eta(2) - real_sol(2)))
print('diferença no t = 2.00001', abs(solver.eta(2.00001) - real_sol(2.00001)))
print('diferença no t = 2.001', abs(solver.eta(2.001) - real_sol(2.001)))
print("max", max(abs(sol - realsol)))


plt.plot(tt, realsol, color="red")
plt.plot(tt, sol, color="blue")
plt.show()
