import random  # For random points, used in test runner

import matplotlib.pyplot as plt
import numpy as np

# Import the adaptive solver and cubic_hermite
from general_delay_rkh import cubic_hermite, solve_dde_adaptive_rk4_hermite

# --- Define the DDE components for this test (Equation 1.2.9) ---


# The discontinuous function f(s)
def f_discontinuous_s(s_val):
    return 1.0 if s_val < 0 else -1.0  # f(s)=1 if s<0, -1 if s>=0


def f_dde_discontinuous_test(t, y, y_delayed):
    """
    DDE function: y'(t) = f(y(t/2)) - y(t)
    f(s) is defined as piecewise: 1 if s<0, -1 if s>=0.
    'y_delayed' is y(t/2).
    """
    return f_discontinuous_s(y_delayed) - y  # y is y(t)


def alpha_func_discontinuous_test(t):
    """Delay function: alpha(t) = t/2."""
    return 0.5 * t


def phi_func_discontinuous_test(t):
    """History function: Y(0) = 1 (implies Y(t)=1 for t <= 0)."""
    return 1.0


# Piecewise analytical solution for y(t) (Equation 1.2.9)
def analytical_sol_discontinuous_test(t):
    # These are the switching points for the analytical solution
    ln2_x2 = 2 * np.log(2)  # Approximately 1.38629
    ln6_x2 = 2 * np.log(6)  # Approximately 3.58352
    ln66_x2 = 2 * np.log(66)  # Approximately 8.44857

    if t < 0:  # History
        return 1.0
    elif t <= ln2_x2 + 1e-9:  # Add tolerance for floating point comparison at boundary
        return 2 * np.exp(-t) - 1
    elif t <= ln6_x2 + 1e-9:
        return 1 - 6 * np.exp(-t)
    elif t <= ln66_x2 + 1e-9:
        return 66 * np.exp(-t) - 1
    else:
        # Analytical solution not provided beyond this point
        return np.nan


# --- Test Runner Function (adapted for adaptive solver) ---
def run_dde_adaptive_test(
    test_name,
    f_dde,
    alpha_func,
    phi_func,
    analytical_sol_func,
    t_span,
    y_initial,
    h_initial,
    TOL,
    max_rejections=10,
    num_random_points_per_interval=10,
    # Parameters for step-size control (from solve_dde_adaptive_rk4_hermite defaults)
    omega_min=0.5,
    omega_max=1.5,
    rho=0.9,
    theta1_val=1 / 3,
    pi1_val=(5 - np.sqrt(5)) / 10,
    pi2_val=(5 + np.sqrt(5)) / 10,
    h_disc_guess=0.1,
    constant_delay_value=None,
):
    """
    Runs an adaptive DDE solver test, including error checks at nodes and random interpolated points.
    This test is expected to fail with a ValueError due to the `get_historical_y` limitation.
    """
    print(f"\n--- Running DDE Adaptive Test: {test_name} ---")
    print(
        f"Time Span: {t_span}, Initial Y: {y_initial}, Initial H: {h_initial}, TOL: {TOL:.1e}"
    )

    try:
        # Call the adaptive solver
        history_data = solve_dde_adaptive_rk4_hermite(
            f_dde,
            alpha_func,
            phi_func,
            t_span,
            y_initial,
            h_initial,
            TOL,
            max_rejections,
            omega_min,
            omega_max,
            rho,
            theta1_val,
            pi1_val,
            pi2_val,
            h_disc_guess,
            constant_delay_value,  # Pass all default/specified args
        )

        # Unpack times and solutions from history_data for testing
        times = [item[0] for item in history_data]
        solutions = np.array([item[1] for item in history_data])
        # Derivatives are in item[2] for m_k/m_kp1 for dense check

        print(f"  Solver completed. Total steps: {len(times)}")

        # --- Calculate Error at nodes and random dense points ---
        max_abs_error_nodes = 0.0
        max_abs_error_dense = 0.0

        # Check error at RK4 nodes (main output points)
        for i in range(len(times)):
            t_val = times[i]
            y_computed = solutions[i]
            y_true = analytical_sol_func(t_val)

            # Handle possible np.nan from analytical_sol_func if t_val is out of range
            if np.isnan(y_true):
                continue

            error = np.max(
                np.abs(y_computed - y_true)
            )  # Max error across components if system
            max_abs_error_nodes = max(max_abs_error_nodes, error)

        # Check error using dense output for random points within intervals
        for i in range(len(history_data) - 1):
            t_k, y_k, m_k = history_data[i]
            t_kp1, y_kp1, m_kp1 = history_data[i + 1]
            h_interval = t_kp1 - t_k

            if h_interval < 1e-12:
                continue  # Skip very tiny or zero-length intervals

            for _ in range(num_random_points_per_interval):
                theta = random.uniform(
                    1e-6, 1.0 - 1e-6
                )  # Random theta strictly between 0 and 1
                t_eval_random = t_k + theta * h_interval

                y_interpolated = cubic_hermite(
                    theta, h_interval, y_k, y_kp1, m_k, m_kp1
                )
                y_true_at_eval = analytical_sol_func(t_eval_random)

                # Handle possible np.nan from analytical_sol_func
                if np.isnan(y_true_at_eval):
                    continue

                error_dense = np.max(np.abs(y_interpolated - y_true_at_eval))
                max_abs_error_dense = max(max_abs_error_dense, error_dense)

        print(f"  Maximum Absolute Error at RK4 nodes: {max_abs_error_nodes:.2e}")
        print(f"  Maximum Absolute Error for dense output: {max_abs_error_dense:.2e}")

        # Test outcome based on tolerance
        if max_abs_error_dense <= TOL:
            print(f"  Test PASSED: Dense output accuracy within tolerance {TOL:.1e}.")
        else:
            print(
                f"  Test FAILED: Dense output error {max_abs_error_dense:.2e} exceeds tolerance {TOL:.1e}."
            )

    except Exception as e:
        # This catch block will execute because of the expected ValueError from get_historical_y
        print(f"  Test FAILED due to an error: {e}")
        print(
            f"  (This error is expected because the `get_historical_y` function cannot currently handle"
        )
        print(
            f"  queries for delayed values that fall within the current step being computed (`t_n < alpha(t_stage) <= t_stage`),"
        )
        print(
            f"  as per previous discussion. This problem (Eq. 1.2.9) triggers that specific limitation.)"
        )

    # --- Plotting the results (optional) ---
    plt.figure(figsize=(10, 6))
    plt.plot(times, solutions, "o", label="Numerical Solution", markersize=4, alpha=0.7)
    # Plot analytical solution only for range where it's defined
    # The range is up to 2*ln(66) for analytical solution, or t_span[1] if it's smaller
    plot_t_analytical_end = min(t_span[1], 2 * np.log(66) + 1e-9)
    t_analytical_plot = np.linspace(t_span[0], plot_t_analytical_end, 500)
    y_analytical_plot = analytical_sol_discontinuous_test(t_analytical_plot)
    plt.plot(
        t_analytical_plot,
        y_analytical_plot,
        "-",
        label="Analytical Solution",
        color="red",
        linewidth=1.5,
    )
    plt.title(r"Solution of $y'(t) = f(y(t/2)) - y(t)$")
    plt.xlabel("t")
    plt.ylabel("y(t)")
    plt.legend()
    plt.grid(True)
    plt.savefig("dde_discontinuous_f_adaptive_solution_plot.png")


# --- Main Test Execution ---
if __name__ == "__main__":
    # This test is expected to fail due to the `get_historical_y` limitation for this problem.
    run_dde_adaptive_test(
        "y'(t) = f(y(t/2)) - y(t) with Discontinuous f - Adaptive Solver Test",
        f_dde_discontinuous_test,
        alpha_func_discontinuous_test,
        phi_func_discontinuous_test,
        analytical_sol_discontinuous_test,
        t_span=(
            0.0,
            2 * np.log(66),
        ),  # Integrate up to the end of the known analytical solution
        y_initial=1.0,
        h_initial=0.0005,  # Initial step size
        TOL=1e-4,  # Tolerance for adaptivity
        max_rejections=10,  # Max rejections per step
        constant_delay_value=None,  # Delay is not constant here
    )
