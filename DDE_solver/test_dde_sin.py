# test_dde_sin.py

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from rkh import solve_dde_rk4_hermite  # Assuming your DDE solver is in rkh.py

# --- Define the DDE components for this test ---
# test set problem 1.1.2


def f_dde_sin_test(t, y, y_delayed):
    """The DDE function f(t, y(t), y(alpha(t)))."""
    return -y_delayed


def alpha_func_sin_test(t):
    """The delay function alpha(t) = t - pi/2."""
    return t - np.pi / 2.0


def phi_func_sin_test(t):
    """The history function phi(t) for t <= t_start."""
    return np.sin(t)  # As per the problem Y(t)=sin(t) for Y<=0


def analytical_sol_sin_test(t):
    """The analytical solution y(t) = sin(t) for t >= 0."""
    return np.sin(t)


# --- DDE Solver Parameters ---
t_start = 0.0
t_end = 4 * np.pi  # Integrate over a few periods to see the oscillation
y_initial = np.sin(t_start)  # y(0) = sin(0) = 0, as per history and analytical solution

# Use a reasonable step size for RK4 and for plotting smoothness
h_step = 0.1
h_disc_guess = 0.01  # Heuristic for discontinuity finder

# --- Main execution block ---
if __name__ == "__main__":
    print(f"--- Running Test: y'(t) = -y(t - pi/2) ---")
    print(f"Integrating from {t_start} to {t_end} with h={h_step}")

    # Solve the DDE
    times, solutions = solve_dde_rk4_hermite(
        f_dde_sin_test,
        alpha_func_sin_test,
        phi_func_sin_test,
        (t_start, t_end),
        y_initial,
        h_step,
        h_disc_guess,
    )

    # --- Calculate Error ---
    max_absolute_error = 0.0
    for t_val, y_val in zip(times, solutions):
        y_true = analytical_sol_sin_test(t_val)
        current_error = np.abs(y_val - y_true)
        max_absolute_error = max(max_absolute_error, current_error)

    print(f"\nMaximum Absolute Error: {max_absolute_error:.2e}")
    # RK4 should be very accurate for this problem and step size
    if max_absolute_error < 1e-4:
        print("Test PASSED: Numerical solution is close to analytical solution.")
    else:
        print("Test FAILED: Numerical solution deviates significantly.")

    # --- Plotting ---
    plt.figure(figsize=(10, 6))

    # Plot numerical solution points
    plt.plot(times, solutions, "o", label="Numerical Solution", markersize=4, alpha=0.7)

    # Plot analytical solution curve for comparison
    t_analytical_plot = np.linspace(t_start, t_end, 500)  # More points for smooth curve
    y_analytical_plot = analytical_sol_sin_test(t_analytical_plot)
    plt.plot(
        t_analytical_plot,
        y_analytical_plot,
        "-",
        label="Analytical Solution (sin(t))",
        color="red",
        linewidth=1.5,
    )

    plt.title(r"Solution of $y'(t) = -y(t - \pi/2)$")
    plt.xlabel("t")
    plt.ylabel("y(t)")
    plt.legend()
    plt.grid(True)

    # --- EASIER SOLUTION: Save the plot to a file ---
    plt.savefig(
        "dde_solution_plot.png"
    )  # You can choose a different filename or format (e.g., .pdf)
    plt.show()  # Remove or comment out this line
