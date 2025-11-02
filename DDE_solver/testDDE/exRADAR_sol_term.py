import numpy as np

from DDE_solver.rkh_refactor import *


def f(t, y, yq):
    return -yq + 5


def phi(t):
    return 9/2 if t < -1 else -1/2


def alpha(t, y):
    return t - 2 - y**2

def exact_sol(t):
    if 0 <= t <= 1:
        return (1/2)*(t-1)
    elif 1 <= t <= 125/121:
        return (11/2)*(t-1)


t_span = [0, 2]
discs = [(-1, 9/2, -1/2)]

Tol = 1e-4
solver = solve_dde(f, alpha, phi, t_span, discs=discs, Atol = Tol, Rtol = Tol)

t_discrete = solver.t
discrete_sol = np.array([exact_sol(t) for t in t_discrete])
discrete_aproxx = np.array(solver.y)
# print('discrete_approx', discrete_aproxx)
# print('discrete_sol', discrete_sol)
# max_discrete_error = np.max(np.abs(discrete_sol - discrete_aproxx))
# print('maximum discrete error', max_discrete_error)



print('Counting steps', Counting.steps)
print('Counting fails', Counting.fails)
print('Counting fcn_calls', Counting.fnc_calls)

plt.plot(solver.t, solver.y, color="blue", label='aproxx')
plt.plot(solver.t, discrete_sol, color="red", label="real sol") 
plt.legend()
plt.show()
