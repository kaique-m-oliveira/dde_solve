import numpy as np
from DDE_solver.rkh_refactor import *


def f(t, y, x, z):
    # print('z', z, '1/z', 1/z)
    return 1 - np.log(1/z)


def phi(t):
    return np.log(t)


def phi_t(t):
    return 1/t


def alpha(t, y):
    return np.exp(1 - (1/t))


def real_sol(t):
    return np.log(t)


def real_sol_t(t):
    return 1/t


epsilon = 0
t_span = [0.1, 2]

solver = solve_dde(f, alpha, phi, t_span, beta=alpha,
                   neutral=True, d_phi=phi_t)


print(f'{'='*80}')
print('PAUL example 2.2.1')
tt = np.linspace(t_span[0], t_span[1], 100)
realsol = np.array([real_sol(t) for t in tt])
sol = [solver.eta(i) for i in tt]
print("max", np.max(np.abs(np.squeeze(sol) - np.squeeze(realsol))))
solution = np.array([real_sol(t) for t in solver.t])
print('adnaed', np.max(np.squeeze(solver.y) - np.squeeze(solution)))


print('sol', len(sol))
print('realsol', len(realsol))
plt.plot(tt, realsol, color="red", label='real solution')
plt.plot(tt, sol, color="blue", label='aproxx')
plt.legend()
plt.show()
