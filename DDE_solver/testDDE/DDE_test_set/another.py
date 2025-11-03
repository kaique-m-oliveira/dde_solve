import numpy as np

from DDE_solver.rkh_refactor import *


def f(t, y, x):
    return 0.2*x/(1+x**10) - 0.1*y

def phi(t):
    return 0.5

def alpha(t, y):
    return t - 14

# No analytical solution found 


t_span = [0, 500]


solver = solve_dde(f, alpha, phi, t_span, Atol=1e-8, Rtol=1e-8)

print(f'{'='*80}')
print(f''' {'='*80} 
      This is problem A1 
      ''')
tt = np.linspace(t_span[0], t_span[1], 100)
sol = [solver.eta(i) for i in tt]

print('==========Counting============')
print('number of steps: ', Counting.steps)
print('number of fails: ', Counting.fails)
print('number of fnc calls: ', Counting.fnc_calls)

plt.plot(tt, sol, color="blue", label='aproxx')
plt.legend()
plt.show()
