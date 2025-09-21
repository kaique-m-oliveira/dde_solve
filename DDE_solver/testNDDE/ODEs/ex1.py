from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import numpy as np


def f(t, y):
    return -y


def exponential_decay(t, y): return -0.5 * y


t_span = [1, 2]
y0 = [1]  # Change y0 to a 1D NumPy array
# sol = solve_ivp(f, t_span, y0, dense_output=True)
sol = solve_ivp(exponential_decay, [0, 10], [2, 4, 8], dense_output=True)


print('sol.t', sol.t, 'shape', sol.t.shape)
print('sol.y', sol.y, 'shape', sol.y.shape)
solution = sol.sol
print('solution', solution)

solution1 = solution.__call__(1),
print('soltion(1)', solution1,  'type', type(solution1))

# Use .T to ensure proper shape for plotting
# plt.plot(sol.t, sol.y, color="red")
# plt.show()
