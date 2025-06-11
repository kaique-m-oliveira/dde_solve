# test_dde_example.py

import numpy as np

# Import your DDE solver function
from rkh import solve_dde_rk4_hermite

# --- Define the DDE for this test ---
# Problem: y'(t) = y(t-1)
# History: y(t) = 1 for t <= 0
# Analytical Solution: For t in [0, 1], y(t) = t + 1.0


def f_dde_test(t, y, y_delayed):
    """The DDE function f(t, y(t), y(alpha(t)))."""
    return y_delayed


def alpha_func_test(t):
    """The delay function alpha(t)."""
    return t - 1.0  # Constant delay of 1


def phi_func_test(t):
    """The history function phi(t) for t <= t_start."""
    return 1.0  # Constant history of 1


# Analytical solution for comparison (only valid for t in [0, 1] in this test)
def analytical_sol_test(t):
    if t <= 1:
        return t + 1.0
    return np.nan  # Return NaN for times outside the simple analytical range


# --- DDE Solver Parameters ---
t_start = 0.0
t_end = 1.0  # Integrate only over the first interval with simple analytical solution
y_initial = 1.0  # y(0) = 1
h_step = 0.1  # Main integration step size
h_disc_guess = 0.01  # Heuristic for discontinuity finder (small for robustness)

# --- Solve the DDE ---
print("--- Running DDE Solver Test Example ---")
print(f"Problem: y'(t) = y(t-1), y(t) = 1 for t <= 0 (Interval: [{t_start}, {t_end}])")

times, solutions = solve_dde_rk4_hermite(
    f_dde_test,
    alpha_func_test,
    phi_func_test,
    (t_start, t_end),
    y_initial,
    h_step,
    h_disc_guess,
)

# --- Print Results and Compare to Analytical Solution ---
print(
    "\nTime           Numerical Solution      Analytical Solution      Absolute Error"
)
print(
    "--------------------------------------------------------------------------------"
)
max_absolute_error = 0.0
for t_val, y_val in zip(times, solutions):
    y_true = analytical_sol_test(t_val)
    current_error = np.abs(y_val - y_true)
    max_absolute_error = max(max_absolute_error, current_error)
    print(f"{t_val:<9.4f} {y_val:<25.6f} {y_true:<25.6f} {current_error:<.2e}")

print(f"\nMaximum Absolute Error over interval: {max_absolute_error:.2e}")

# Check for a typical RK4 accuracy given the step size
if max_absolute_error < 1e-4:  # A common tolerance for RK4 with h=0.1
    print("Test PASSED: Numerical solution is close to analytical solution.")
else:
    print(
        "Test FAILED: Numerical solution deviates significantly from analytical solution."
    )
