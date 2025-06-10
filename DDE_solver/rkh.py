#     t_start, t_end = t_span
#
#     # 1. Pre-compute discontinuity points
#     # Use h_step for the heuristic in find_discontinuity_chain (or h_disc_guess)
#     discs_found = find_discontinuity_chain(alpha_func, t_span, h_disc_guess, 500)
#     grid_points = sorted(
#         list(set([t_start] + [d for d in discs_found if t_start <= d <= t_end]))
#     )
#
#     # 2. History storage: list of (t, y, y_prime) tuples for Hermite interpolation
#     history_data = []
#
#     # 3. Helper function (closure) to get y(t_query) from history or phi
#     def get_historical_y(t_query):
#         if t_query <= t_start:  # Query is in the initial history segment
#             return phi_func(t_query)
#
#         # Search for the interval [t_k, t_k+1] in history_data
#         for i in range(len(history_data) - 1):
#             t_k, y_k, m_k = history_data[i]
#             t_kp1, y_kp1, m_kp1 = history_data[i + 1]
#
#             if (
#                 t_k <= t_query <= t_kp1 + 1e-9 * (t_kp1 - t_k)
#             ):  # Add tolerance for FP at boundary
#                 theta = (t_query - t_k) / (t_kp1 - t_k)
#                 return cubic_hermite(theta, t_kp1 - t_k, y_k, y_kp1, m_k, m_kp1)
#
#         # Should not happen for alpha(t) <= t and correct step sizing.
#         # Indicates query is outside processed history or in an unhandled current-step interpolation case.
#         raise ValueError(
#             f"History query t={t_query} outside processed history or t_start. History up to {history_data[-1][0] if history_data else 'None'}."
#         )
#
#     # --- Initialize the first point in history_data (t_start, y_initial, y'_initial) ---
#     y_current = np.asarray(y_initial)
#     alpha_t_start_val = alpha_func(t_start)
#     y_delayed_at_start = np.asarray(
#         get_historical_y(alpha_t_start_val)
#     )  # Uses phi_func if alpha(t_start) <= t_start
#     m_current = np.asarray(f_dde(t_start, y_current, y_delayed_at_start))  # y'(t_start)
#
#     history_data.append((t_start, y_current, m_current))
#
#     times_out = [t_start]
#     solutions_out = [y_current]
#
#     # --- Main Integration Loop ---
#     current_grid_idx = 0
#     while t_current < t_end:
#         t_current = history_data[-1][0]  # Current time from last history point
#         y_current = history_data[-1][1]  # Current solution
#
#         if t_current >= t_end:
#             break  # Just in case loop conditions miss exact t_end
#
#         # Determine the next target time for integration (min of h_step or next discontinuity)
#         t_target_candidate = t_current + h_step
#
#         # Check if a pre-computed grid point (discontinuity) is next
#         if current_grid_idx + 1 < len(grid_points):
#             next_disc_in_grid = grid_points[current_grid_idx + 1]
#             if (
#                 next_disc_in_grid <= t_target_candidate + 1e-9 * h_step
#             ):  # Hit disc exactly or very close
#                 t_target_candidate = next_disc_in_grid
#                 current_grid_idx += 1  # Advance grid point index
#             # else: current_grid_idx remains as next_disc_in_grid is not reached this step
#
#         t_next_node = min(t_target_candidate, t_end)  # Final target time for this step
#
#         h_actual = t_next_node - t_current
#         if (
#             h_actual < 1e-12
#         ):  # Avoid tiny steps or infinite loops (e.g., if already at t_end)
#             break
#
#         # Perform one RK4 step
#         y_next = rk4_step_dde(
#             f_dde, alpha_func, get_historical_y, t_current, y_current, h_actual
#         )
#
#         # Calculate derivative at the new point (m_next) for history storage
#         alpha_t_next_val = alpha_func(t_next_node)
#         y_delayed_at_next = np.asarray(get_historical_y(alpha_t_next_val))
#         m_next = np.asarray(f_dde(t_next_node, y_next, y_delayed_at_next))
#
#         # Add the new (t, y, m) point to history
#         history_data.append((t_next_node, y_next, m_next))
#
#         # Store solution for output
#         times_out.append(t_next_node)
#         solutions_out.append(y_next)
#
#     return times_out, solutions_out
#
#
# # --- Example DDE Usage (for testing) ---
# if __name__ == "__main__":
#     # Define a simple DDE: y'(t) = -y(t) + y(t-1)
#     # y(t) = 1 for t <= 0
#     # Expected behavior: Smooth growth or decay after t=0.
#     def f_dde_example(t, y, y_delayed):
#         return -y + y_delayed
#
#     def alpha_func_example(t):
#         return t - 1.0  # Constant delay of 1
#
#     def phi_func_example(t):
#         return 1.0  # History function
#
#     # --- Run the DDE solver ---
#     t_start_dde = 0.0
#     t_end_dde = 5.0
#     y_initial_dde = 1.0  # Initial condition at t_start
#     h_step_dde = 0.25  # Main integration step size
#     h_disc_guess_dde = 0.1  # Heuristic for finding discontinuities
#
#     print("Solving DDE: y'(t) = -y(t) + y(t-1), y(t)=1 for t<=0")
#     times_dde, solutions_dde = solve_dde_rk4_hermite(
#         f_dde_example,
#         alpha_func_example,
#         phi_func_example,
#         (t_start_dde, t_end_dde),
#         y_initial_dde,
#         h_step_dde,
#         h_disc_guess_dde,
#     )
#
#     print("\n--- DDE Solution Output ---")
#     print("Time           Solution")
#     print("-------------------------")
#     for t, y in zip(times_dde, solutions_dde):
#         # Expected exact solution y(t) = 1 (for t <= 1)
#         # Then it gets more complex, e.g., y(t) = 2-t for t in [1,2]
#         print(f"{t:<15.4f} {y:<15.6f}")
#
#     # Example for a system DDE
#     # Let y = [y1, y2]. y1'(t) = y2(t) + y1(t-0.5), y2'(t) = -y1(t) + y2(t-0.5)
#     # alpha(t) = [t-0.5, t-0.5] (vector output, same delay)
#     # phi(t) = [cos(t), sin(t)] for t <= 0
#     def f_dde_system_example(t, y, y_delayed):
#         y1, y2 = y
#         y1_d, y2_d = y_delayed
#         return np.array([y2 + y1_d, -y1 + y2_d])
#
#     def alpha_func_system_example(t):
#         return np.array([t - 0.5, t - 0.5])  # Returns a vector
#
#     def phi_func_system_example(t):
#         return np.array([np.cos(t), np.sin(t)])
#
#     print("\n--- Solving System DDE ---")
#     t_start_sys = 0.0
#     t_end_sys = 2.0
#     y_initial_sys = np.array([1.0, 0.0])  # y1(0)=1, y2(0)=0
#     h_step_sys = 0.1
#     h_disc_guess_sys = 0.05
#
#     times_sys, solutions_sys = solve_dde_rk4_hermite(
#         f_dde_system_example,
#         alpha_func_system_example,
#         phi_func_system_example,
#         (t_start_sys, t_end_sys),
#         y_initial_sys,
#         h_step_sys,
#         h_disc_guess_sys,
#     )
#
#     print("Time           Solution (y1, y2)")
#     print("---------------------------------")
#     for t, y_vec in zip(times_sys, solutions_sys):
#         print(f"{t:<15.4f} {str(np.round(y_vec, 6)):<25}")
#
#################################################################################################################
import numpy as np
from scipy.optimize import root

# # --- Reusing previously defined concise functions ---
#
#
# # rk4_step_dde: y'(t) = f(t, y(t), y(alpha(t)))
# # Assumes f_dde(t, y_curr, y_delayed)
# # Assumes alpha_func(t)
# # Assumes get_historical_y_wrapper(t_eval)
# def rk4_step_dde(f_dde, alpha_func, get_historical_y_wrapper, t_n, y_n, h):
#     y_n_arr = np.asarray(y_n)
#     t1 = t_n
#     y_d1 = np.asarray(get_historical_y_wrapper(alpha_func(t1)))
#     k1 = np.asarray(f_dde(t1, y_n_arr, y_d1))
#     t2 = t_n + 0.5 * h
#     y_arg2 = y_n_arr + 0.5 * h * k1
#     y_d2 = np.asarray(get_historical_y_wrapper(alpha_func(t2)))
#     k2 = np.asarray(f_dde(t2, y_arg2, y_d2))
#     t3 = t_n + 0.5 * h
#     y_arg3 = y_n_arr + 0.5 * h * k2
#     y_d3 = np.asarray(get_historical_y_wrapper(alpha_func(t3)))
#     k3 = np.asarray(f_dde(t3, y_arg3, y_d3))
#     t4 = t_n + h
#     y_arg4 = y_n_arr + h * k3
#     y_d4 = np.asarray(get_historical_y_wrapper(alpha_func(t4)))
#     k4 = np.asarray(f_dde(t4, y_arg4, y_d4))
#     return y_n_arr + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
#
#
# # cubic_hermite: (theta, h_interval, y0, y1, m0, m1)
# def cubic_hermite(theta, h_interval, y0, y1, m0, m1):
#     y0_arr, y1_arr = np.asarray(y0), np.asarray(y1)
#     m0_arr, m1_arr = np.asarray(m0), np.asarray(m1)
#     t2, t3 = theta * theta, theta * theta * theta
#     h00 = 2 * t3 - 3 * t2 + 1
#     h10 = t3 - 2 * t2 + theta
#     h01 = -2 * t3 + 3 * t2
#     h11 = t3 - t2
#     return (
#         h00 * y0_arr
#         + h01 * y1_arr
#         + h10 * h_interval * m0_arr
#         + h11 * h_interval * m1_arr
#     )
#
#
# # find_discontinuity_chain (as last provided, adapted to new variable names)
# def find_discontinuity_chain(
#     alpha_func, t_span_find_disc, h_disc_find, max_iterations_find_disc
# ):
#     t0_find, tf_find = t_span_find_disc
#     discs = [t0_find]
#     tk = t0_find
#     try:
#         check_t = tk + h_disc_find
#         if check_t <= tk:
#             check_t = tk + 1e-6
#         test_alpha_output_arr = np.asarray(alpha_func(check_t))
#         num_components = test_alpha_output_arr.size
#         if num_components == 0:
#             return [t0_find]
#     except Exception as e:
#         # print(f"Error in find_discontinuity_chain initial check: {e}") # For debugging
#         return [t0_find]
#
#     for _ in range(max_iterations_find_disc):
#         if tk >= tf_find:
#             break
#         candidate_next_roots = []
#         for j in range(num_components):
#             eq = lambda tt: np.asarray(alpha_func(tt)).flat[j] - tk
#             guess = tk + 10 * h_disc_find
#             if guess <= tk:
#                 guess = tk + 1e-6
#             try:
#                 result = root(eq, guess)
#                 if result.success:
#                     next_t_cand = result.x[0]
#                     if next_t_cand > tk + 1e-9 and next_t_cand <= tf_find + 1e-9:
#                         candidate_next_roots.append(next_t_cand)
#             except Exception:
#                 pass
#         if not candidate_next_roots:
#             break
#         tk = np.min(candidate_next_roots)
#         discs.append(tk)
#
#     return discs
#
#
# # --- Main DDE Solver Implementation ---
#
#
# def solve_dde_rk4_hermite(
#     f_dde, alpha_func, phi_func, t_span, y_initial, h_step, h_disc_guess=0.1
# ):
#     """
#     Solves a DDE using RK4 with cubic Hermite interpolation and discontinuity tracking.
#
#     y'(t) = f_dde(t, y(t), y(alpha(t)))
#     y(t) = phi_func(t) for t <= t_span[0]
#
#     Args:
#         f_dde (callable): DDE function f(t, y_current, y_delayed_value).
#         alpha_func (callable): Delay function alpha(t) returning delayed time.
#         phi_func (callable): History function phi(t) for t <= t_span[0].
#         t_span (tuple): (t_start, t_end) for the integration interval.
#         y_initial (float or np.ndarray): Initial value y(t_start).
#         h_step (float): Main integration step size.
#         h_disc_guess (float, optional): Heuristic step for discontinuity finding. Defaults to 0.1.
#
#     Returns:
#         tuple: (times_out, solutions_out) lists of time points and solutions.
#     """

# --- Reusing previously defined concise functions ---


# rk4_step_dde: y'(t) = f(t, y(t), y(alpha(t)))
def rk4_step_dde(f_dde, alpha_func, get_historical_y_wrapper, t_n, y_n, h):
    y_n_arr = np.asarray(y_n)
    t1 = t_n
    y_d1 = np.asarray(get_historical_y_wrapper(alpha_func(t1)))
    k1 = np.asarray(f_dde(t1, y_n_arr, y_d1))
    t2 = t_n + 0.5 * h
    y_arg2 = y_n_arr + 0.5 * h * k1
    y_d2 = np.asarray(get_historical_y_wrapper(alpha_func(t2)))
    k2 = np.asarray(f_dde(t2, y_arg2, y_d2))
    t3 = t_n + 0.5 * h
    y_arg3 = y_n_arr + 0.5 * h * k2
    y_d3 = np.asarray(get_historical_y_wrapper(alpha_func(t3)))
    k3 = np.asarray(f_dde(t3, y_arg3, y_d3))
    t4 = t_n + h
    y_arg4 = y_n_arr + h * k3
    y_d4 = np.asarray(get_historical_y_wrapper(alpha_func(t4)))
    k4 = np.asarray(f_dde(t4, y_arg4, y_d4))
    return y_n_arr + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


# cubic_hermite: (theta, h_interval, y0, y1, m0, m1)
def cubic_hermite(theta, h_interval, y0, y1, m0, m1):
    y0_arr, y1_arr = np.asarray(y0), np.asarray(y1)
    m0_arr, m1_arr = np.asarray(m0), np.asarray(m1)
    t2, t3 = theta * theta, theta * theta * theta
    h00 = 2 * t3 - 3 * t2 + 1
    h10 = t3 - 2 * t2 + theta
    h01 = -2 * t3 + 3 * t2
    h11 = t3 - t2
    return (
        h00 * y0_arr
        + h01 * y1_arr
        + h10 * h_interval * m0_arr
        + h11 * h_interval * m1_arr
    )


# find_discontinuity_chain (as last provided)
def find_discontinuity_chain(
    alpha_func, t_span_find_disc, h_disc_find, max_iterations_find_disc
):
    t0_find, tf_find = t_span_find_disc
    discs = [t0_find]
    tk = t0_find
    try:  # Robustness for initial check of alpha_func return type
        check_t = tk + h_disc_find
        if check_t <= tk:
            check_t = tk + 1e-6
        test_alpha_output_arr = np.asarray(alpha_func(check_t))
        num_components = test_alpha_output_arr.size
        if num_components == 0:
            return [t0_find]
    except Exception as e:
        return [
            t0_find
        ]  # For debugging: print(f"Error in find_discontinuity_chain initial check: {e}")

    for _ in range(max_iterations_find_disc):
        if tk >= tf_find:
            break
        candidate_next_roots = []
        for j in range(num_components):
            eq = lambda tt: np.asarray(alpha_func(tt)).flat[j] - tk
            guess = tk + 10 * h_disc_find
            if guess <= tk:
                guess = tk + 1e-6
            try:
                result = root(eq, guess)
                if result.success:
                    next_t_cand = result.x[0]
                    if next_t_cand > tk + 1e-9 and next_t_cand <= tf_find + 1e-9:
                        candidate_next_roots.append(next_t_cand)
            except Exception:
                pass
        if not candidate_next_roots:
            break
        next_tk = np.min(candidate_next_roots)
        discs.append(next_tk)
        tk = next_tk
    return discs


# --- Main DDE Solver Implementation ---


def solve_dde_rk4_hermite(
    f_dde, alpha_func, phi_func, t_span, y_initial, h_step, h_disc_guess=0.1
):
    """
    Solves a DDE using RK4 with cubic Hermite interpolation and discontinuity tracking.

    y'(t) = f_dde(t, y(t), y(alpha(t)))
    y(t) = phi_func(t) for t <= t_span[0]
    """
    t_start, t_end = t_span

    # 1. Pre-compute discontinuity points
    discs_found = find_discontinuity_chain(alpha_func, t_span, h_disc_guess, 500)
    grid_points = sorted(
        list(set([t_start] + [d for d in discs_found if t_start <= d <= t_end]))
    )

    # 2. History storage: list of (t, y, y_prime) tuples for Hermite interpolation
    history_data = []

    # 3. Helper function (closure) to get y(t_query) from history or phi
    def get_historical_y(t_query):
        if t_query <= t_start:  # Query is in the initial history segment
            return phi_func(t_query)

        # Search for the interval [t_k, t_k+1] in history_data
        for i in range(len(history_data) - 1):
            t_k, y_k, m_k = history_data[i]
            t_kp1, y_kp1, m_kp1 = history_data[i + 1]

            if (
                t_k <= t_query <= t_kp1 + 1e-9 * (t_kp1 - t_k)
            ):  # Add tolerance for FP at boundary
                theta = (t_query - t_k) / (t_kp1 - t_k)
                return cubic_hermite(theta, t_kp1 - t_k, y_k, y_kp1, m_k, m_kp1)

        # If t_query is outside processed history (should ideally not happen for alpha(t) <= t)
        raise ValueError(
            f"get_historical_y: Query time {t_query} outside processed history or t_start. History up to {history_data[-1][0] if history_data else 'None'}."
        )

    # --- Initialize the first point in history_data (t_start, y_initial, y'_initial) ---
    y_current_val = np.asarray(y_initial)
    alpha_t_start_val = alpha_func(t_start)
    y_delayed_at_start = np.asarray(get_historical_y(alpha_t_start_val))
    m_current_val = np.asarray(f_dde(t_start, y_current_val, y_delayed_at_start))

    history_data.append((t_start, y_current_val, m_current_val))

    times_out = [t_start]
    solutions_out = [y_current_val]

    # --- Main Integration Loop ---
    current_grid_idx = 0

    # Initialize t_current and y_current from the first point in history_data
    # These will be the starting state for the first RK4 step.
    t_current = history_data[0][0]  # t_start
    y_current = history_data[0][1]  # y_initial

    while t_current < t_end:  # Now t_current is properly bound
        # Determine the next target time (min of h_step or next discontinuity)
        t_target_candidate = t_current + h_step

        # Check if a pre-computed grid point (discontinuity) is next
        if current_grid_idx + 1 < len(grid_points):
            next_disc_in_grid = grid_points[current_grid_idx + 1]
            if (
                next_disc_in_grid <= t_target_candidate + 1e-9 * h_step
            ):  # Hit disc exactly or very close
                t_target_candidate = next_disc_in_grid
                current_grid_idx += 1  # Advance grid point index
            # else: current_grid_idx remains, next_disc_in_grid is not reached this step

        t_next_node = min(t_target_candidate, t_end)  # Final target time for this step

        h_actual = t_next_node - t_current
        if h_actual < 1e-12:  # Avoid tiny steps or infinite loops if already at t_end
            break

        # Perform one RK4 step
        y_next = rk4_step_dde(
            f_dde, alpha_func, get_historical_y, t_current, y_current, h_actual
        )

        # Calculate derivative at the new point (m_next) for history storage
        alpha_t_next_val = alpha_func(t_next_node)
        y_delayed_at_next = np.asarray(get_historical_y(alpha_t_next_val))
        m_next = np.asarray(f_dde(t_next_node, y_next, y_delayed_at_next))

        # Add the new (t, y, m) point to history
        history_data.append((t_next_node, y_next, m_next))

        # Store solution for output
        times_out.append(t_next_node)
        solutions_out.append(y_next)

        # Update t_current and y_current for the *next* iteration of the loop
        t_current = t_next_node
        y_current = y_next

    # return times_out, solutions_out
    return times_out, solutions_out, history_data  # ADDED history_data to the return
