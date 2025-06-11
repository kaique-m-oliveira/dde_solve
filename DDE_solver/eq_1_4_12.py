import random  # For random points

import numpy as np
from rkh import cubic_hermite  # Import cubic_hermite for dense output check
from rkh import solve_dde_rk4_hermite

# --- Define the DDE components for this test (as before) ---


def f_dde_system_sin_cos_test(t, y, y_delayed):
    """
    The DDE function for a system y'(t) = -y(t - pi/2).
    'y' and 'y_delayed' are expected to be NumPy arrays.
    """
    return np.array([-y_delayed[0], -y_delayed[1]])


def alpha_func_system_sin_cos_test(t):
    """The delay function alpha(t) = t - pi/2 (scalar output)."""
    return t - np.pi / 2.0


def phi_func_system_sin_cos_test(t):
    """The history function Y(t) = [sin(t), cos(t)]^T for t <= t_start."""
    return np.array([np.sin(t), np.cos(t)])


def analytical_sol_system_sin_cos_test(t):
    """The analytical solution y(t) = [sin(t), cos(t)]^T for t >= t_start."""
    return np.array([np.sin(t), np.cos(t)])


# --- Test Runner Function ---
def run_dde_test_no_plot(
    test_name,
    f_dde,
    alpha_func,
    phi_func,
    analytical_sol_func,
    t_span,
    y_initial,
    h_step,
    h_disc_guess,
    tolerance=1e-5,
    num_random_points_per_interval=10,
):
    """
    Runs a DDE test, performs RK4 integration, and checks accuracy at random
    points within each interval using Hermite interpolation, without plotting.
    """
    print(f"\n--- Running DDE Test: {test_name} ---")
    print(f"Time Span: {t_span}, Initial Y: {y_initial}, Step H: {h_step}")
    print(
        f"Tolerance for dense output: {tolerance:.1e}, Random points per interval: {num_random_points_per_interval}"
    )

    try:
        # Solve the DDE and capture history_data directly
        history_data = solve_dde_rk4_hermite(  # MODIFIED: Capture history_data directly
            f_dde, alpha_func, phi_func, t_span, y_initial, h_step, h_disc_guess
        )

        # Unpack times and solutions from history_data for compatibility with test logic
        times = [item[0] for item in history_data]
        solutions = [item[1] for item in history_data]
        # Derivatives (m) are also in history_data as item[2], but not needed directly for this test's output.

        print(f"  Solver completed. Total main steps (nodes): {len(times)}")
        print(
            f"  Total history data points (t, y, y'): {len(history_data)}"
        )  # This count is now accurate

        max_abs_error_nodes = 0.0  # Error at the main RK4 computed nodes
        max_abs_error_dense = 0.0  # Error at the random interpolated points

        # --- Check accuracy at RK4 nodes (main output points) ---
        for i in range(len(times)):
            t_val = times[i]
            y_computed = solutions[i]
            y_true = analytical_sol_func(t_val)
            error = np.max(np.abs(y_computed - y_true))  # Max error across components
            max_abs_error_nodes = max(max_abs_error_nodes, error)

        # --- Check accuracy using dense output for random points within intervals ---
        # Iterate through history_data directly, as it contains (t,y,m) tuples
        for i in range(len(history_data) - 1):
            t_k, y_k, m_k = history_data[i]  # Start point data
            t_kp1, y_kp1, m_kp1 = history_data[i + 1]  # End point data
            h_interval = t_kp1 - t_k  # Length of the current interval

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

                error_dense = np.max(np.abs(y_interpolated - y_true_at_eval))
                max_abs_error_dense = max(max_abs_error_dense, error_dense)

        print(f"  Maximum Absolute Error at RK4 nodes: {max_abs_error_nodes:.2e}")
        print(f"  Maximum Absolute Error for dense output: {max_abs_error_dense:.2e}")

        # Final check against tolerance
        if max_abs_error_dense <= tolerance:
            print(
                f"  Test PASSED: Dense output accuracy within tolerance {tolerance:.1e}."
            )
        else:
            print(
                f"  Test FAILED: Dense output error {max_abs_error_dense:.2e} exceeds tolerance {tolerance:.1e}."
            )

    except Exception as e:
        print(f"  Test FAILED due to an error: {e}")


# --- Main Test Execution ---
if __name__ == "__main__":
    # Parameters for the system DDE (y'(t) = -y(t - pi/2))
    run_dde_test_no_plot(
        "System DDE y'(t) = -y(t - pi/2) - No Plot",
        f_dde_system_sin_cos_test,
        alpha_func_system_sin_cos_test,
        phi_func_system_sin_cos_test,
        analytical_sol_system_sin_cos_test,
        t_span=(np.pi / 2.0, np.pi / 2.0 + 4 * np.pi),
        y_initial=np.array([1.0, 0.0]),  # y(pi/2) = [sin(pi/2), cos(pi/2)]
        h_step=0.01,  # Smaller h_step for more points and accuracy
        h_disc_guess=0.01,
        tolerance=1e-4,
        num_random_points_per_interval=10,
    )


# # test_dde_system_sin_cos_no_plot.py
#
# import random  # For random points
#
# import numpy as np
# from rkh import cubic_hermite  # Import cubic_hermite for dense output check
# from rkh import solve_dde_rk4_hermite
#
# # --- Define the DDE components for this test ---
#
#
# def f_dde_system_sin_cos_test(t, y, y_delayed):
#     """
#     The DDE function for a system y'(t) = -y(t - pi/2).
#     'y' and 'y_delayed' are expected to be NumPy arrays.
#     """
#     return np.array([-y_delayed[0], -y_delayed[1]])
#
#
# def alpha_func_system_sin_cos_test(t):
#     """The delay function alpha(t) = t - pi/2 (scalar output)."""
#     return t - np.pi / 2.0
#
#
# def phi_func_system_sin_cos_test(t):
#     """The history function Y(t) = [sin(t), cos(t)]^T for t <= t_start."""
#     return np.array([np.sin(t), np.cos(t)])
#
#
# def analytical_sol_system_sin_cos_test(t):
#     """The analytical solution y(t) = [sin(t), cos(t)]^T for t >= t_start."""
#     return np.array([np.sin(t), np.cos(t)])
#
#
# # --- Test Runner Function ---
# def run_dde_test_no_plot(
#     test_name,
#     f_dde,
#     alpha_func,
#     phi_func,
#     analytical_sol_func,
#     t_span,
#     y_initial,
#     h_step,
#     h_disc_guess,
#     tolerance=1e-5,
#     num_random_points_per_interval=10,
# ):
#     """
#     Runs a DDE test, performs RK4 integration, and checks accuracy at random
#     points within each interval using Hermite interpolation, without plotting.
#     """
#     print(f"\n--- Running DDE Test: {test_name} ---")
#     print(f"Time Span: {t_span}, Initial Y: {y_initial}, Step H: {h_step}")
#     print(
#         f"Tolerance for dense output: {tolerance:.1e}, Random points per interval: {num_random_points_per_interval}"
#     )
#
#     try:
#         # Solve the DDE and capture the history_data for dense output checks
#         times, solutions, history_data = solve_dde_rk4_hermite(
#             f_dde, alpha_func, phi_func, t_span, y_initial, h_step, h_disc_guess
#         )
#         print(f"  Solver completed. Total main steps (nodes): {len(times)}")
#         print(f"  Total history data points (t, y, y'): {len(history_data)}")
#
#         max_abs_error_nodes = 0.0  # Error at the main RK4 computed nodes
#         max_abs_error_dense = 0.0  # Error at the random interpolated points
#
#         # --- Check accuracy at RK4 nodes (main output points) ---
#         for i in range(len(times)):
#             t_val = times[i]
#             y_computed = solutions[i]
#             y_true = analytical_sol_func(t_val)
#             error = np.max(np.abs(y_computed - y_true))  # Max error across components
#             max_abs_error_nodes = max(max_abs_error_nodes, error)
#
#         # --- Check accuracy using dense output for random points within intervals ---
#         for i in range(len(history_data) - 1):
#             t_k, y_k, m_k = history_data[i]  # Start point data
#             t_kp1, y_kp1, m_kp1 = history_data[i + 1]  # End point data
#             h_interval = t_kp1 - t_k  # Length of the current interval
#
#             if h_interval < 1e-12:
#                 continue  # Skip very tiny or zero-length intervals
#
#             for _ in range(num_random_points_per_interval):
#                 theta = random.uniform(
#                     1e-6, 1.0 - 1e-6
#                 )  # Random theta strictly between 0 and 1
#                 t_eval_random = t_k + theta * h_interval
#
#                 y_interpolated = cubic_hermite(
#                     theta, h_interval, y_k, y_kp1, m_k, m_kp1
#                 )
#                 y_true_at_eval = analytical_sol_func(t_eval_random)
#
#                 error_dense = np.max(np.abs(y_interpolated - y_true_at_eval))
#                 max_abs_error_dense = max(max_abs_error_dense, error_dense)
#
#         print(f"  Maximum Absolute Error at RK4 nodes: {max_abs_error_nodes:.2e}")
#         print(f"  Maximum Absolute Error for dense output: {max_abs_error_dense:.2e}")
#
#         # Final check against tolerance
#         if max_abs_error_dense <= tolerance:
#             print(
#                 f"  Test PASSED: Dense output accuracy within tolerance {tolerance:.1e}."
#             )
#         else:
#             print(
#                 f"  Test FAILED: Dense output error {max_abs_error_dense:.2e} exceeds tolerance {tolerance:.1e}."
#             )
#
#     except Exception as e:
#         print(f"  Test FAILED due to an error: {e}")
#
#
# # --- Main Test Execution ---
# if __name__ == "__main__":
#     # Parameters for the system DDE (y'(t) = -y(t - pi/2))
#     run_dde_test_no_plot(
#         "System DDE y'(t) = -y(t - pi/2) - No Plot",
#         f_dde_system_sin_cos_test,
#         alpha_func_system_sin_cos_test,
#         phi_func_system_sin_cos_test,
#         analytical_sol_system_sin_cos_test,
#         t_span=(np.pi / 2.0, np.pi / 2.0 + 4 * np.pi),
#         y_initial=np.array([1.0, 0.0]),  # y(pi/2) = [sin(pi/2), cos(pi/2)]
#         h_step=0.01,
#         h_disc_guess=0.01,
#         tolerance=1e-4,
#         num_random_points_per_interval=10,
#     )
