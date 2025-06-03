import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp


def rk4_system(f, y0, t0, t_end, h):
    """
    Solves a system of ODEs using the Runge-Kutta 4 method.

    Parameters:
    - f: function representing the system of ODEs, f(t, y),
         where y is a numpy array.
    - y0: initial values, a numpy array with the same length as the system of ODEs.
    - t0: initial time.
    - t_end: end time.
    - h: step size.

    Returns:
    - t_values: numpy array of time points.
    - y_values: numpy array of solution values at each time point.
    """
    t_values = np.arange(t0, t_end + h, h)
    y_values = np.zeros((len(t_values), len(y0)))
    y_values[0] = y0
    print(y_values)

    for i in range(1, len(t_values)):
        t = t_values[i - 1]
        y = y_values[i - 1]
        print("-" * 99)
        print("last", y)

        k1 = h * f(t, y)
        k2 = h * f(t + h / 2, y + k1 / 2)
        k3 = h * f(t + h / 2, y + k2 / 2)
        k4 = h * f(t + h, y + k3)

        y_values[i] = y + (k1 + 2 * k2 + 2 * k3 + k4) / 6

    return t_values, y_values


# Define a system of differential equations, e.g., dy1/dt = -y2, dy2/dt = y1.
def system(t, y):
    dy1 = -y[1]
    dy2 = y[0]
    return np.array([dy1, dy2])


# Initial conditions and parameters
y0 = [1.0, 0.0]  # Initial values for y1 and y2
t0 = 0.0  # Initial time
t_end = 10.0  # End time
solution = solve_ivp(
    system, (t0, t_end), y0, method="RK45", t_eval=np.linspace(t0, t_end, 100)
)

# Plotting both solutions
plt.figure(figsize=(10, 5))
plt.plot(
    solution.t,
    solution.y[0],
    label="$y_1(t)$ (solve_ivp)",
    linestyle="--",
    color="blue",
)
plt.plot(
    solution.t,
    solution.y[1],
    label="$y_2(t)$ (solve_ivp)",
    linestyle="--",
    color="orange",
)

# Solution from our RK4 implementation
t_values, y_values = rk4_system(system, y0, t0, t_end, h=0.1)
plt.plot(t_values, y_values[:, 0], label="$y_1(t)$ (RK4)", color="blue")
plt.plot(t_values, y_values[:, 1], label="$y_2(t)$ (RK4)", color="orange")

# Configure plot
plt.xlabel("Time $t$")
plt.ylabel("Solution $y(t)$")
plt.title("Comparison of RK4 and solve_ivp Solutions")
plt.legend()
plt.grid(True)
plt.show()
