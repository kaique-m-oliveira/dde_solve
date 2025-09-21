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


def f(t, y, yq):
    return -yq


def phi(t):
    return 1


def alpha(t, y):
    return t - 1


def real_sol(t):
    if 0 <= t <= 1:
        return 1 - t
    if 1 <= t <= 2:
        return (1/2)*(t**2 - 4*t + 3)
    if 2 <= t <= 3:
        return (1/6) * (17 - 24*t + 9*t**2 - t**3)
    return 0


t_span = [0, 3]

d_f = [1, lambda t, y, x: [0], lambda t, y, x: [-1]]
d_alpha = [lambda t, y: [1], lambda t, y: [0]]
def d_phi(t): return 0


# solver = Solver(f, alpha, phi, t_span, d_f, d_alpha, d_phi)
solver = solve_dde(f, alpha, phi, t_span, d_f, d_alpha, d_phi)

# solver.f_y = lambda t, y, x: 0
# solver.f_x = lambda t, y, x: -1
# solver.alpha_t = lambda t, y: 1
# solver.alpha_y = lambda t, y: 0
# solver.phi_t = lambda t: 0
# solver.etas_t.append(lambda t: 0)

print(f'{'='*80}')
print('ex 1_1_1_zenaro.py')
tt = np.linspace(t_span[0], t_span[1], 100)
realsol = np.array([real_sol(t) for t in tt])
sol = [solver.eta(i) for i in tt]
# for i in range(len(tt)):
#     print(tt[i], realsol[i] - sol[i])
print("max", np.max(np.abs(np.squeeze(sol) - np.squeeze(realsol))))
solution = np.array([real_sol(t) for t in solver.t])
print('solution', solution)
print('shape solver.y', solver.y)
print('adnaed', np.max(np.squeeze(solver.y) - np.squeeze(solution)))


print('sol', len(sol))
print('realsol', len(realsol))
plt.plot(tt, realsol, color="red", label='real solution')
plt.plot(tt, sol, color="blue", label='aproxx')
plt.legend()
plt.show()
