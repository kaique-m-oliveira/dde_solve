import numpy as np
from DDE_solver.rkh_refactor import *


# def f(t, y, x, z):
#     return -4*t*y**2 / 4 + np.log(np.cos(2*t))**2 + np.tan(2*t) + 0.5*np.arctan(z)
def f(t, y, x, z):
    return -(4 * t * y**2) / (4 + (np.log(np.cos(2*t)))**2) + np.tan(2*t) + 0.5 * np.arctan(z)


def phi(t):
    return 0


def phi_t(t):
    return 0


def alpha(t, y):
    return t*y**2 / (1 + y**2)


def real_sol(t):
    return -np.log(np.cos(2*t))/2


t_span = [0, 0.225*np.pi]

solver = solve_dde(f, alpha, phi, t_span, beta=alpha,
                   neutral=True, d_phi=phi_t)


print(f'{'='*80}')
print('JULIA NDDE vanishing delay?')
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
