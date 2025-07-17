from DDE_solver.rkh_state import *

# WARN: STATE EXAMPLE


def f(t, y, yq):
    return np.exp(yq)/t


def phi(t):
    return 0


def alpha(t, y):
    return y - np.log(2) + 1


def real_sol(t):
    return np.log(t) if 1 <= t <= 2 else 0.5 * t + np.log(2) - 1


t_span = [1, 4]

solver = Solver(f, alpha, phi, t_span)
solver.solve_dde(discs=[2])

tt = np.linspace(t_span[0], t_span[1] - 0.05, 100)
realsol = np.array([real_sol(t) for t in tt])
sol = np.array([solver.eta(i) for i in tt])
# for i in range(len(tt)):
#     print(tt[i], realsol[i] - sol[i])
#
print('t values', solver.t)
print('diferença no t = 1.999', abs(solver.eta(1.999) - real_sol(1.999)))
print('diferença no t = 2', abs(solver.eta(2) - real_sol(2)))
print('diferença no t = 2.00001', abs(solver.eta(2.00001) - real_sol(2.00001)))
print('diferença no t = 2.001', abs(solver.eta(2.001) - real_sol(2.001)))
print("max", max(abs(sol - realsol)))


plt.plot(tt, realsol, color="red")
plt.plot(tt, sol, color="blue")
plt.show()
