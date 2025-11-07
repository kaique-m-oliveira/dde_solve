import numpy as np

from DDE_solver.rkh_refactor import *


def f(t, y, yq):
    return 1 - yq


def phi(t):
    return np.log(t)


def alpha(t, y):
    return np.exp(1 - (1/t))


def real_sol(t):
    return np.log(t)


t_span = [0.5, 5]

print(f'{'='*80}')
print('ex_1_2_7.py')

solver = solve_dde(f, alpha, phi, t_span, Atol=1e-12, Rtol = 1e-12)

tt = np.linspace(t_span[0], t_span[1], 10000)
realsol = np.array([real_sol(t) for t in tt])
sol = np.array([solver.eta(i) for i in tt])
sol_processed = np.squeeze(sol)
max_error = np.max(np.abs(sol_processed - realsol))
y = np.squeeze(solver.y)
solution = np.array([np.squeeze(real_sol(t)) for t in solver.t])
print('max_error', max_error)

plt.plot(tt, realsol, color="red", label='real solution')
plt.plot(tt, sol, color="blue", label='aproxx')
plt.legend()
plt.show()
