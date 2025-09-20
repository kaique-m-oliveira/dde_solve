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
    return yq


def phi(t):
    return 1


def alpha(t, y):
    return t - y


t_span = [0, 10]

solver = solve_dde(f, alpha, phi, t_span)


print(f'{'='*80}')
print('ex 1_1_1_zenaro.py')
# tt = np.linspace(t_span[0], t_span[1], 100)
tt = np.linspace(-5, t_span[1], 100)
sol = [solver.eta(i) for i in tt]
print('shape solver.y', solver.y)


print('sol', len(sol))
plt.plot(tt, sol, color="blue", label='aproxx')
plt.plot(tt, tt + 1, color="red", label='real_sol')
plt.legend()
plt.show()
