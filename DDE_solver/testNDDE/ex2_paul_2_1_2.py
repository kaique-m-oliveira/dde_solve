# from DDE_solver.rkh_state import *
# from DDE_solver.rkh_step_rejection import *
# from DDE_solver.rkh_testing import *
# from DDE_solver.rkh import *
import numpy as np
# from DDE_solver.rkh_vectorize import *
# from DDE_solver.rkh_multiple_delays import *
from DDE_solver.rkh_refactor import *
# from DDE_solver.rkh_refactor_before_chatgpt import *
# from DDE_solver.rkh_state_complete import *
# from DDE_solver.rkh_NDDE import *

# WARN: STATE EXAMPLE


def f(t, y, x, z):
    return y + x - 2 * z


def phi(t):
    if t <= 0:
        return -t
    # if 0 <= t <= 1:
    #     return -2 + t + 2*np.exp(t)


def d_phi(t):
    if t <= 0:
        return -1
    # if 0 <= t <= 1:
    #     return 1 + 2*np.exp(t)


def alpha(t, y):
    return t - 1


def real_sol(t):
    if 0 <= t <= 1:
        return -2 + t + 2*np.exp(t)
    elif 1 <= t <= 2:
        return 4 - t + 2*np.exp(t) - 2*(t + 1)*np.exp(t - 1)


t_span = [0, 2]


solver = solve_dde(f, alpha, phi, t_span, beta=alpha,
                   neutral=True, d_phi=d_phi)

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
