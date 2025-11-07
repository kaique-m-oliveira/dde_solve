import numpy as np

from DDE_solver.rkh_refactor import *


def f(t, y, x, z):
    return np.cos(t)*(1 + x) + 0.6 * y* z


def phi(t):
    return -t/2


def phi_t(t):
    return -1/2


def alpha(t, y):
    return t*y**2


epsilon = 0
t_span = [0.25, 5]


Tol = 1e-9
solver = solve_ndde(t_span, f, alpha, alpha, phi, phi_t, method='RKC5', discs=[], Atol=Tol, Rtol=Tol)



print(f'{'='*80}')
print('paper radar computing breaking points')
tt = np.linspace(t_span[0], t_span[1], 100)
sol = [solver.eta(i) for i in tt]
sol_t = [solver.eta_t(i) for i in tt]

plt.plot(tt, sol, color="blue", label='eta')
plt.plot(tt, sol_t, color="red", label='eta_t')
plt.legend()
plt.show()
