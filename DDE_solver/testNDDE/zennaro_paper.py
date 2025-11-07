import numpy as np

from DDE_solver.rkh_refactor import *


def f(t, y, x, z):
    return -y/2 - z


def phi(t):
    return 1 - t


def phi_t(t):
    return -1


def alpha(t, y):
    return (t - 0.5)*y**2


epsilon = 0
t_span = [0, 4]

Tol = 1e-8
solver = solve_ndde(t_span, f, alpha, alpha, phi, phi_t, method='RKC5', Atol=Tol, Rtol=Tol)

# solver = solve_ndde(f, alpha, phi, t_span, beta=alpha,
#                    neutral=True, d_phi=phi_t)
print('steps', Counting.steps)
print('rejection', Counting.fails)
print('func calls', Counting.fnc_calls)


print(f'{'='*80}')
print('paper zennaro: NUMERICS FOR NEUTRAL DELAY DIFFERENTIAL EQUATIONS')
tt = np.linspace(t_span[0], t_span[1], 100)
sol = [solver.eta(i) for i in tt]

plt.plot(tt, sol, color="blue", label='aproxx')
plt.legend()
plt.show()
