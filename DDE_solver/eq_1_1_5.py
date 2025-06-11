# test_dde_cos_sin.py

import matplotlib.pyplot as plt
import numpy as np
from rkh import solve_dde_rk4_hermite  # Assuming your DDE solver is in rkh.py

# --- Define the DDE components for this test ---


def f_dde_cos_sin_test(t, y, y_delayed):
    """The DDE function y'(t) = y(t) + y(t-pi) + 3cos(t) + 5sin(t)."""
    return y + y_delayed + 3 * np.cos(t) + 5 * np.sin(t)


def alpha_func_cos_sin_test(t):
    """The delay function alpha(t) = t - pi."""
    return t - np.pi


def phi_func_cos_sin_test(t):
    """The history function y(t) = 3sin(t) - 5cos(t) for t <= t_start."""
    return 3 * np.sin(t) - 5 * np.cos(t)


def analytical_sol_cos_sin_test(t):
    """The analytical solution y(t) = 3sin(t) - 5cos(t) for t >= 0."""
    return 3 * np.sin(t) - 5 * np.cos(t)


# --- DDE Solver Parameters ---
t_start = 0.0
t_end = 4 * np.pi
y_initial = phi_func_cos_sin_test(t_start)

# --- CHANGE THIS LINE ---
h_step = 0.01  # Try a much smaller step size
# --- END CHANGE ---

h_disc_guess = 0.01


# --- Main execution block ---
if __name__ == "__main__":
    print(f"--- Running Test: y'(t) = y(t) + y(t-pi) + 3cos(t) + 5sin(t) ---")
    print(f"Integrating from {t_start} to {t_end} with h={h_step}")

    # Solve the DDE
    times, solutions, history = solve_dde_rk4_hermite(
        f_dde_cos_sin_test,
        alpha_func_cos_sin_test,
        phi_func_cos_sin_test,
        (t_start, t_end),
        y_initial,
        h_step,
        h_disc_guess,
    )

    # --- Calculate Error ---
    max_absolute_error = 0.0
    for t_val, y_val in zip(times, solutions):
        y_true = analytical_sol_cos_sin_test(t_val)
        current_error = np.abs(y_val - y_true)
        max_absolute_error = max(max_absolute_error, current_error)

    print(f"\nMaximum Absolute Error: {max_absolute_error:.2e}")
    # RK4 should be very accurate for this problem and step size
    if max_absolute_error < 1e-4:
        print("Test PASSED: Numerical solution is close to analytical solution.")
    else:
        print("Test FAILED: Numerical solution deviates significantly.")

    # --- Plotting the results ---
    plt.figure(figsize=(10, 6))

    # Plot numerical solution points
    plt.plot(times, solutions, "o", label="Numerical Solution", markersize=4, alpha=0.7)

    # Plot analytical solution curve for comparison
    t_analytical_plot = np.linspace(t_start, t_end, 500)  # More points for smooth curve
    y_analytical_plot = analytical_sol_cos_sin_test(t_analytical_plot)
    plt.plot(
        t_analytical_plot,
        y_analytical_plot,
        "-",
        label="Analytical Solution",
        color="red",
        linewidth=1.5,
    )

    plt.title(r"Solution of $y'(t) = y(t) + y(t-\pi) + 3\cos(t) + 5\sin(t)$")
    plt.xlabel("t")
    plt.ylabel("y(t)")
    plt.legend()
    plt.grid(True)

    # Save the plot to a file
    plt.savefig("dde_cos_sin_solution_plot.png")
