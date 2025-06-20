import bisect  # Make sure this import is at the top of your rkh.py file

import numpy as np
from scipy.optimize import root

# ... (rest of rk4_step_dde, cubic_hermite, find_discontinuity_chain, solve_dde_rk4_hermite's outer structure) ...


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


# NEW: Specialized constant delay discontinuity finder (from previous turn)
def find_discontinuity_chain_constant_delay(tau_constant, t_span, max_discs=2000):
    t0, tf = t_span
    discs = []
    if tau_constant <= 0:
        return [t0]
    current_t_disc = t0
    discs.append(current_t_disc)
    for _ in range(max_discs):
        current_t_disc += tau_constant
        if current_t_disc <= tf + 1e-9:  # Add tolerance for comparison
            discs.append(current_t_disc)
        else:
            break
    return discs


# General discontinuity finder (from previous turn, used if delay is not constant)
def find_discontinuity_chain_general(
    alpha_func, t_span_find_disc, h_disc_find, max_iterations_find_disc
):
    t0_find, tf_find = t_span_find_disc
    discs = [t0_find]
    tk = t0_find
    try:
        check_t = tk + h_disc_find
        if check_t <= tk:
            check_t = tk + 1e-6
        test_alpha_output_arr = np.asarray(alpha_func(check_t))
        num_components = test_alpha_output_arr.size
        if num_components == 0:
            return [t0_find]
    except Exception:
        return [t0_find]

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


def rk4_step_dde(
    f_dde,
    alpha_func,
    get_historical_y_wrapper,
    t_n,
    y_n,
    h,
    # NEW OPTIMIZATION CONTEXT PARAMETERS
    current_grid_idx_at_tn=0,
    k_delay_steps_val=0,
    is_constant_delay_opt_active_flag=False,
):
    """
    Performs one RK4 step for a DDE, passing context for get_historical_y optimization.
    """
    y_n_arr = np.asarray(y_n)

    # Note: t1, t2, t3, t4 are the "t_stage_current_RK" for get_historical_y
    # current_grid_idx_at_tn is the "current_grid_idx_for_opt"
    # k_delay_steps_val is "k_steps"
    # is_constant_delay_opt_active_flag is "is_opt_active"

    t1 = t_n
    y_d1 = np.asarray(
        get_historical_y_wrapper(
            alpha_func(t1),
            t1,
            current_grid_idx_at_tn,
            is_constant_delay_opt_active_flag,
            k_delay_steps_val,
        )
    )
    k1 = np.asarray(f_dde(t1, y_n_arr, y_d1))

    t2 = t_n + 0.5 * h
    y_arg2 = y_n_arr + 0.5 * h * k1
    y_d2 = np.asarray(
        get_historical_y_wrapper(
            alpha_func(t2),
            t2,
            current_grid_idx_at_tn,
            is_constant_delay_opt_active_flag,
            k_delay_steps_val,
        )
    )
    k2 = np.asarray(f_dde(t2, y_arg2, y_d2))

    t3 = t_n + 0.5 * h
    y_arg3 = y_n_arr + 0.5 * h * k2
    y_d3 = np.asarray(
        get_historical_y_wrapper(
            alpha_func(t3),
            t3,
            current_grid_idx_at_tn,
            is_constant_delay_opt_active_flag,
            k_delay_steps_val,
        )
    )
    k3 = np.asarray(f_dde(t3, y_arg3, y_d3))

    t4 = t_n + h
    y_arg4 = y_n_arr + h * k3
    y_d4 = np.asarray(
        get_historical_y_wrapper(
            alpha_func(t4),
            t4,
            current_grid_idx_at_tn + 1,
            is_constant_delay_opt_active_flag,
            k_delay_steps_val,
        )
    )  # Index for t_n+h is current_grid_idx_at_tn + 1
    k4 = np.asarray(f_dde(t4, y_arg4, y_d4))

    return y_n_arr + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


# --- (cubic_hermite, find_discontinuity_chain_constant_delay, find_discontinuity_chain_general remain the same) ---


def solve_dde_rk4_hermite(
    f_dde,
    alpha_func,
    phi_func,
    t_span,
    y_initial,
    h_step,
    h_disc_guess=0.1,
    constant_delay_value=None,
):
    t_start, t_end = t_span

    # --- Step 1: Adjust h_step for constant delay if applicable ---
    h_actual_for_calc = h_step
    k_delay_steps_val = 0  # Integer 'k' for tau = k*h
    is_constant_delay_opt_active_flag = False
    if constant_delay_value is not None and constant_delay_value > 0:
        is_constant_delay_opt_active_flag = True
        k_val = int(round(constant_delay_value / h_step))
        if k_val <= 0:
            k_val = 1
        h_actual_for_calc = constant_delay_value / k_val
        k_delay_steps_val = k_val
        print(
            f"INFO: Adjusted h_step from {h_step} to {h_actual_for_calc:.6f} for constant delay optimization (k={k_delay_steps_val})."
        )

    # --- Step 2: Choose and execute discontinuity finder ---
    if constant_delay_value is not None:
        discs_found = find_discontinuity_chain_constant_delay(
            constant_delay_value, t_span
        )
    else:
        discs_found = find_discontinuity_chain_general(
            alpha_func, t_span, h_disc_guess, 500
        )

    grid_points = sorted(
        list(set([t_start] + [d for d in discs_found if t_start <= d <= t_end]))
    )

    # --- Step 3: History storage and helper (get_historical_y) ---
    history_data = []  # Stores (t, y, m) tuples
    history_times_only = []  # Stores only time points for efficient binary search

    # get_historical_y now takes additional arguments for optimization context
    def get_historical_y(
        t_query,
        t_stage_current_RK=None,
        current_grid_idx_at_tn=None,
        is_opt_active=False,
        k_steps=0,
    ):
        if t_query <= t_start:
            return phi_func(t_query)

        # --- NEW OPTIMIZATION LOGIC: Direct lookup for exact mesh points (skipping bisect) ---
        # This optimization is active only if the constant delay optimization is on,
        # and if t_stage_current_RK is an exact mesh point (t_n or t_n+h_actual).
        if is_opt_active and current_grid_idx_at_tn is not None and k_steps > 0:
            # Check if t_stage_current_RK corresponds to t_n (start of interval)
            if np.isclose(
                t_stage_current_RK, history_times_only[current_grid_idx_at_tn]
            ):
                target_idx = current_grid_idx_at_tn - k_steps
                if target_idx >= 0 and np.isclose(
                    t_query, history_times_only[target_idx]
                ):
                    return history_data[target_idx][1]
            # Check if t_stage_current_RK corresponds to t_n + h_actual (end of interval)
            # This applies when get_historical_y is called for k4 and final m_next calculations
            elif current_grid_idx_at_tn + 1 < len(history_times_only) and np.isclose(
                t_stage_current_RK, history_times_only[current_grid_idx_at_tn + 1]
            ):
                target_idx = (current_grid_idx_at_tn + 1) - k_steps
                if target_idx >= 0 and np.isclose(
                    t_query, history_times_only[target_idx]
                ):
                    return history_data[target_idx][1]
            # If the optimization is active but t_stage_current_RK is not an exact grid point (e.g., t_n + 0.5h),
            # or the direct lookup failed, then fall back to bisect/interpolation.

        # --- Fallback to binary search for general cases (non-optimized or intermediate RK stages) ---
        idx = bisect.bisect_right(history_times_only, t_query)

        # Direct lookup if t_query is an exact mesh point (already handled here, but relies on bisect)
        if idx > 0 and np.isclose(t_query, history_times_only[idx - 1]):
            return history_data[idx - 1][1]

        # If not an exact mesh point, but within a known interval, interpolate
        if idx > 0 and idx < len(history_data):
            t_k, y_k, m_k = history_data[idx - 1]
            t_kp1, y_kp1, m_kp1 = history_data[idx]

            if t_k <= t_query <= t_kp1 + 1e-9 * (t_kp1 - t_k):
                theta = (t_query - t_k) / (t_kp1 - t_k)
                return cubic_hermite(theta, t_kp1 - t_k, y_k, y_kp1, m_k, m_kp1)

        # Fallback (query in current step being computed, or out of bounds)
        raise ValueError(
            f"get_historical_y: Query time {t_query} outside processed history or t_start. History up to {history_data[-1][0] if history_data else 'None'}."
        )

    # --- Step 4: Initialize history and main integration loop ---
    y_current_val = np.asarray(y_initial)
    alpha_t_start_val = alpha_func(t_start)
    # Initial call to get_historical_y for m_current_val (no special optimization context needed here)
    y_delayed_at_start = np.asarray(get_historical_y(alpha_t_start_val))
    m_current_val = np.asarray(f_dde(t_start, y_current_val, y_delayed_at_start))

    history_data.append((t_start, y_current_val, m_current_val))
    history_times_only.append(t_start)

    current_grid_idx = 0
    t_current = t_start
    y_current = y_current_val

    while t_current < t_end:
        t_target_candidate = t_current + h_actual_for_calc

        if current_grid_idx + 1 < len(grid_points):
            next_disc_in_grid = grid_points[current_grid_idx + 1]
            if next_disc_in_grid <= t_target_candidate + 1e-9 * h_actual_for_calc:
                t_target_candidate = next_disc_in_grid
                current_grid_idx += 1

        t_next_node = min(t_target_candidate, t_end)

        h_actual = t_next_node - t_current
        if h_actual < 1e-12:
            break

        # Call rk4_step_dde with optimization context for get_historical_y
        y_next = rk4_step_dde(
            f_dde,
            alpha_func,
            get_historical_y,
            t_current,
            y_current,
            h_actual,
            current_grid_idx,
            k_delay_steps_val,
            is_constant_delay_opt_active_flag,
        )

        alpha_t_next_val = alpha_func(t_next_node)
        # Call get_historical_y for m_next with optimization context (t_next_node is a mesh point)
        y_delayed_at_next = np.asarray(
            get_historical_y(
                alpha_t_next_val,
                t_next_node,
                current_grid_idx + 1,  # current_grid_idx + 1 is index of t_next_node
                is_constant_delay_opt_active_flag,
                k_delay_steps_val,
            )
        )
        m_next = np.asarray(f_dde(t_next_node, y_next, y_delayed_at_next))

        history_data.append((t_next_node, y_next, m_next))
        history_times_only.append(t_next_node)

        t_current = t_next_node
        y_current = y_next

    return history_data

    # --- (cubic_hermite, find_discontinuity_chain_constant_delay, find_discontinuity_chain_general remain the same) ---

    # WARN: this version works but is not optimized for constant delays
    # # find_discontinuity_chain (as last provided)
    # def find_discontinuity_chain(
    #     alpha_func, t_span_find_disc, h_disc_find, max_iterations_find_disc
    # ):
    #     t0_find, tf_find = t_span_find_disc
    #     discs = [t0_find]
    #     tk = t0_find
    #     try:  # Robustness for initial check of alpha_func return type
    #         check_t = tk + h_disc_find
    #         if check_t <= tk:
    #             check_t = tk + 1e-6
    #         test_alpha_output_arr = np.asarray(alpha_func(check_t))
    #         num_components = test_alpha_output_arr.size
    #         if num_components == 0:
    #             return [t0_find]
    #     except Exception as e:
    #         return [
    #             t0_find
    #         ]  # For debugging: print(f"Error in find_discontinuity_chain initial check: {e}")
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
    #         next_tk = np.min(candidate_next_roots)
    #         discs.append(next_tk)
    #         tk = next_tk
    #     return discs
    #
    #
    # def find_discontinuity_chain_constant_delay(tau_constant, t_span, max_discs=2000):
    #     """
    #     Finds discontinuity points for a DDE with a constant delay tau_constant.
    #     Points are t_0, t_0 + tau, t_0 + 2*tau, ... up to t_f.
    #
    #     Args:
    #         tau_constant (float): The constant delay value (must be > 0 for propagation).
    #         t_span (tuple): (t_start, t_end) for the integration interval.
    #         max_discs (int, optional): Maximum number of discontinuities to find. Safety limit.
    #                                   Defaults to 2000.
    #
    #     Returns:
    #         list: Sorted list of discontinuity points within t_span.
    #     """
    #     t0, tf = t_span
    #     discs = []
    #
    #     if (
    #         tau_constant <= 0
    #     ):  # If delay is non-positive, no propagation in this simple form
    #         return [t0]  # Only the initial discontinuity is added
    #
    #     current_t_disc = t0
    #     discs.append(current_t_disc)  # Initial discontinuity (t_start)
    #
    #     # Generate subsequent discontinuities by adding tau_constant
    #     for _ in range(max_discs):  # Loop up to max_discs for safety
    #         current_t_disc += tau_constant
    #         if (
    #             current_t_disc <= tf + 1e-9
    #         ):  # Add a small tolerance for floating point comparison
    #             discs.append(current_t_disc)
    #         else:
    #             break  # Beyond t_end, stop
    #
    #     return discs  # Already sorted due to sequential generation
    #
    #
    # # --- (rk4_step_dde, cubic_hermite, find_discontinuity_chain remain the same) ---
    #
    # # --- (rk4_step_dde, cubic_hermite, find_discontinuity_chain remain the same) ---
    # # Paste find_discontinuity_chain_constant_delay here or ensure it's imported.
    #
    #
    # def solve_dde_rk4_hermite(
    #     f_dde,
    #     alpha_func,
    #     phi_func,
    #     t_span,
    #     y_initial,
    #     h_step,
    #     h_disc_guess=0.1,
    #     constant_delay_value=None,  # NEW OPTIONAL PARAMETER
    # ):
    #     """
    #     Solves a DDE using RK4 with cubic Hermite interpolation and discontinuity tracking.
    #
    #     y'(t) = f_dde(t, y(t), y(alpha(t)))
    #     y(t) = phi_func(t) for t <= t_span[0]
    #
    #     Args:
    #         ... (existing args) ...
    #         constant_delay_value (float, optional): If the delay is constant, provide its value here (tau).
    #                                                This activates an optimized discontinuity finder.
    #                                                Defaults to None (uses general finder).
    #
    #     Returns:
    #         list: A list of (t, y, y_prime) tuples, representing the solution history.
    #     """
    #     t_start, t_end = t_span
    #
    #     # 1. Choose and execute discontinuity finder
    #     if constant_delay_value is not None:
    #         # Use optimized constant delay finder
    #         discs_found = find_discontinuity_chain_constant_delay(
    #             constant_delay_value, t_span
    #         )
    #     else:
    #         # Use general (scipy.optimize.root based) discontinuity finder
    #         discs_found = find_discontinuity_chain(alpha_func, t_span, h_disc_guess, 500)
    #
    #     grid_points = sorted(
    #         list(set([t_start] + [d for d in discs_found if t_start <= d <= t_end]))
    #     )
    #
    #     # 2. History storage: list of (t, y, y_prime) tuples
    #     history_data = []
    #
    #     # Inside solve_dde_rk4_hermite, this is the definition of get_historical_y:
    #     def get_historical_y(t_query):
    #         # 1. Query is in the initial history segment (t <= t_start)
    #         if t_query <= t_start:
    #             return phi_func(t_query)
    #
    #         # 2. Query is within a fully processed history interval (t_k <= t <= t_k+1 for t_k+1 <= t_n)
    #         # Use binary search (bisect_right) for efficient lookup.
    #         history_times = [item[0] for item in history_data]
    #
    #         idx = bisect.bisect_right(history_times, t_query)
    #
    #         if idx > 0 and idx < len(history_data):
    #             t_k, y_k, m_k = history_data[idx - 1]
    #             t_kp1, y_kp1, m_kp1 = history_data[idx]
    #
    #             if t_k <= t_query <= t_kp1 + 1e-9 * (t_kp1 - t_k):
    #                 theta = (t_query - t_k) / (t_kp1 - t_k)
    #                 return cubic_hermite(theta, t_kp1 - t_k, y_k, y_kp1, m_k, m_kp1)
    #
    #         # This ValueError is raised for queries within the *current* step being computed,
    #         # or if t_query is somehow outside the expected range of history_data.
    #         raise ValueError(
    #             f"get_historical_y: Query time {t_query} outside processed history or t_start. History up to {history_data[-1][0] if history_data else 'None'}."
    #         )

    # WARN: old linear search one
    # 3. Helper function (closure) to get y(t_query) from history or phi
    # def get_historical_y(t_query):
    #     if t_query <= t_start:
    #         return phi_func(t_query)
    #
    #     for i in range(len(history_data) - 1):
    #         t_k, y_k, m_k = history_data[i]
    #         t_kp1, y_kp1, m_kp1 = history_data[i + 1]
    #
    #         # if t_k <= t_query <= t_kp1 + 1e-9 * (t_kp1 - t_k): #before
    #         if t_k <= t_query <= t_kp1:
    #             theta = (t_query - t_k) / (t_kp1 - t_k)
    #             return cubic_hermite(theta, t_kp1 - t_k, y_k, y_kp1, m_k, m_kp1)
    #
    #     raise ValueError(
    #         f"get_historical_y: Query time {t_query} outside processed history or t_start. History up to {history_data[-1][0] if history_data else 'None'}."
    #     )

    # --- Initialize the first point in history_data (t_start, y_initial, y'_initial) ---
    y_current_val = np.asarray(y_initial)
    alpha_t_start_val = alpha_func(t_start)
    y_delayed_at_start = np.asarray(get_historical_y(alpha_t_start_val))
    m_current_val = np.asarray(f_dde(t_start, y_current_val, y_delayed_at_start))

    history_data.append((t_start, y_current_val, m_current_val))

    # --- Main Integration Loop ---
    current_grid_idx = 0
    t_current = t_start
    y_current = y_current_val

    while t_current < t_end:
        t_target_candidate = t_current + h_step

        if current_grid_idx + 1 < len(grid_points):
            next_disc_in_grid = grid_points[current_grid_idx + 1]
            if next_disc_in_grid <= t_target_candidate + 1e-9 * h_step:
                t_target_candidate = next_disc_in_grid
                current_grid_idx += 1

        t_next_node = min(t_target_candidate, t_end)

        h_actual = t_next_node - t_current
        if h_actual < 1e-12:
            break

        y_next = rk4_step_dde(
            f_dde, alpha_func, get_historical_y, t_current, y_current, h_actual
        )

        alpha_t_next_val = alpha_func(t_next_node)
        y_delayed_at_next = np.asarray(get_historical_y(alpha_t_next_val))
        m_next = np.asarray(f_dde(t_next_node, y_next, y_delayed_at_next))

        history_data.append((t_next_node, y_next, m_next))

        t_current = t_next_node
        y_current = y_next

    return history_data


# WARN: this was the last version
# def solve_dde_rk4_hermite(
#     f_dde, alpha_func, phi_func, t_span, y_initial, h_step, h_disc_guess=0.1
# ):
#     """
#     Solves a DDE using RK4 with cubic Hermite interpolation and discontinuity tracking.
#
#     y'(t) = f_dde(t, y(t), y(alpha(t)))
#     y(t) = phi_func(t) for t <= t_span[0]
#
#     Returns:
#         list: A list of (t, y, y_prime) tuples, representing the solution history.
#               t is the time point, y is the solution vector/scalar,
#               and y_prime is the derivative approximation at that point.
#     """
#     t_start, t_end = t_span
#
#     # 1. Pre-compute discontinuity points
#     discs_found = find_discontinuity_chain(alpha_func, t_span, h_disc_guess, 500)
#     grid_points = sorted(
#         list(set([t_start] + [d for d in discs_found if t_start <= d <= t_end]))
#     )
#
#     # 2. History storage: list of (t, y, y_prime) tuples for Hermite interpolation
#     # This list will directly be the output.
#     history_data = []
#
#     # 3. Helper function (closure) to get y(t_query) from history or phi
#     def get_historical_y(t_query):
#         if t_query <= t_start:
#             return phi_func(t_query)
#
#         for i in range(len(history_data) - 1):
#             t_k, y_k, m_k = history_data[i]
#             t_kp1, y_kp1, m_kp1 = history_data[i + 1]
#
#             if t_k <= t_query <= t_kp1 + 1e-9 * (t_kp1 - t_k):
#                 theta = (t_query - t_k) / (t_kp1 - t_k)
#                 return cubic_hermite(theta, t_kp1 - t_k, y_k, y_kp1, m_k, m_kp1)
#
#         raise ValueError(
#             f"get_historical_y: Query time {t_query} outside processed history or t_start. History up to {history_data[-1][0] if history_data else 'None'}."
#         )
#
#     # --- Initialize the first point in history_data (t_start, y_initial, y'_initial) ---
#     y_current_val = np.asarray(y_initial)
#     alpha_t_start_val = alpha_func(t_start)
#     y_delayed_at_start = np.asarray(get_historical_y(alpha_t_start_val))
#     m_current_val = np.asarray(f_dde(t_start, y_current_val, y_delayed_at_start))
#
#     # Append the initial state to history_data
#     history_data.append((t_start, y_current_val, m_current_val))
#
#     # --- Main Integration Loop ---
#     current_grid_idx = 0
#     t_current = t_start
#     y_current = y_current_val
#
#     while t_current < t_end:
#         t_target_candidate = t_current + h_step
#
#         if current_grid_idx + 1 < len(grid_points):
#             next_disc_in_grid = grid_points[current_grid_idx + 1]
#             if next_disc_in_grid <= t_target_candidate + 1e-9 * h_step:
#                 t_target_candidate = next_disc_in_grid
#                 current_grid_idx += 1
#
#         t_next_node = min(t_target_candidate, t_end)
#
#         h_actual = t_next_node - t_current
#         if h_actual < 1e-12:
#             break
#
#         y_next = rk4_step_dde(
#             f_dde, alpha_func, get_historical_y, t_current, y_current, h_actual
#         )
#
#         alpha_t_next_val = alpha_func(t_next_node)
#         y_delayed_at_next = np.asarray(get_historical_y(alpha_t_next_val))
#         m_next = np.asarray(f_dde(t_next_node, y_next, y_delayed_at_next))
#
#         # Add the new (t, y, m) point to history_data directly
#         history_data.append((t_next_node, y_next, m_next))
#
#         # Update t_current and y_current for the next iteration
#         t_current = t_next_node
#         y_current = y_next
#
#     return history_data  # MODIFIED RETURN: Now directly returns the list of (t, y, m) tuples


# --- (rk4_step_dde, cubic_hermite, find_discontinuity_chain remain the same) ---
# WARN: SECOND ATTEMPT, NOT GOOD ENOUGH

# def solve_dde_rk4_hermite(
#     f_dde, alpha_func, phi_func, t_span, y_initial, h_step, h_disc_guess=0.1
# ):
#     """
#     Solves a DDE using RK4 with cubic Hermite interpolation and discontinuity tracking.
#
#     y'(t) = f_dde(t, y(t), y(alpha(t)))
#     y(t) = phi_func(t) for t <= t_span[0]
#
#     Returns:
#         tuple: (times_out, solutions_out, m_out)
#             times_out (list): Time points.
#             solutions_out (list): Solution values at those time points.
#             m_out (list): Derivative approximations (y') at those time points.
#     """
#     t_start, t_end = t_span
#
#     # 1. Pre-compute discontinuity points
#     discs_found = find_discontinuity_chain(alpha_func, t_span, h_disc_guess, 500)
#     grid_points = sorted(
#         list(set([t_start] + [d for d in discs_found if t_start <= d <= t_end]))
#     )
#
#     # 2. History storage: list of (t, y, y_prime) tuples for Hermite interpolation
#     history_data = []
#     m_out = []  # NEW: List to store derivative approximations for output
#
#     # 3. Helper function (closure) to get y(t_query) from history or phi
#     def get_historical_y(t_query):
#         if t_query <= t_start:
#             return phi_func(t_query)
#
#         for i in range(len(history_data) - 1):
#             t_k, y_k, m_k = history_data[i]
#             t_kp1, y_kp1, m_kp1 = history_data[i + 1]
#
#             if t_k <= t_query <= t_kp1 + 1e-9 * (t_kp1 - t_k):
#                 theta = (t_query - t_k) / (t_kp1 - t_k)
#                 return cubic_hermite(theta, t_kp1 - t_k, y_k, y_kp1, m_k, m_kp1)
#
#         raise ValueError(
#             f"get_historical_y: Query time {t_query} outside processed history or t_start. History up to {history_data[-1][0] if history_data else 'None'}."
#         )
#
#     # --- Initialize the first point in history_data (t_start, y_initial, y'_initial) ---
#     y_current_val = np.asarray(y_initial)
#     alpha_t_start_val = alpha_func(t_start)
#     y_delayed_at_start = np.asarray(get_historical_y(alpha_t_start_val))
#     m_current_val = np.asarray(f_dde(t_start, y_current_val, y_delayed_at_start))
#
#     history_data.append((t_start, y_current_val, m_current_val))
#
#     times_out = [t_start]
#     solutions_out = [y_current_val]
#     m_out.append(m_current_val)  # NEW: Add initial derivative to m_out
#
#     # --- Main Integration Loop ---
#     current_grid_idx = 0
#     t_current = t_start
#     y_current = y_current_val
#
#     while t_current < t_end:
#         t_target_candidate = t_current + h_step
#
#         if current_grid_idx + 1 < len(grid_points):
#             next_disc_in_grid = grid_points[current_grid_idx + 1]
#             if next_disc_in_grid <= t_target_candidate + 1e-9 * h_step:
#                 t_target_candidate = next_disc_in_grid
#                 current_grid_idx += 1
#
#         t_next_node = min(t_target_candidate, t_end)
#
#         h_actual = t_next_node - t_current
#         if h_actual < 1e-12:
#             break
#
#         y_next = rk4_step_dde(
#             f_dde, alpha_func, get_historical_y, t_current, y_current, h_actual
#         )
#
#         alpha_t_next_val = alpha_func(t_next_node)
#         y_delayed_at_next = np.asarray(get_historical_y(alpha_t_next_val))
#         m_next = np.asarray(f_dde(t_next_node, y_next, y_delayed_at_next))
#
#         history_data.append((t_next_node, y_next, m_next))
#
#         times_out.append(t_next_node)
#         solutions_out.append(y_next)
#         m_out.append(m_next)  # NEW: Add m_next to m_out
#
#         t_current = t_next_node
#         y_current = y_next
#
#     return times_out, solutions_out, m_out  # MODIFIED RETURN


# --- Main DDE Solver Implementation ---


# def solve_dde_rk4_hermite(
#     f_dde, delay, phi_func, t_span, y_initial, h_step, h_guess=0.1
# ):
#     """
#     Solves a DDE using RK4 with cubic Hermite interpolation and discontinuity tracking.
#
#     y'(t) = f_dde(t, y(t), y(alpha(t)))
#     y(t) = phi_func(t) for t <= t_span[0]
#     """
#     t_0, t_f = t_span
#
#     # Getting discs before hand
#     discs = find_discontinuity_chain(delay, t_span, h_guess, 500)
#     grid_points = sorted(list(set([t_0] + [d for d in discs if t_0 <= d <= t_f])))
#
#     # Linear search for tt \in [t_k, t_k+1], then return the value interpolated
#     history = []
#
#     def get_history(tt):
#         if tt <= t_0:
#             return phi_func(tt)
#
#         for i in range(len(history) - 1):
#             tk, yk, mk = history[i]
#             tk_next, yk_next, mk_next = history[i + 1]
#
#             # if ( tk <= tt <= tk1) + 1e-9 * (tk1 - tk): #maybe useless
#             if tk <= tt <= tk_next:
#                 theta = (tt - tk) / (tk_next - tk)
#                 return cubic_hermite(theta, tk_next - tk, yk, yk_next, mk, mk_next)
#
#         raise ValueError(f"{tt} >= {tk_next} and we can¬t  deal with that")
#
#     # --- Initialize the first point in history_data (t_start, y_initial, y'_initial) ---
#     y_current_val = np.asarray(y_initial)
#     alpha_t_start_val = delay(t_0)
#     y_delayed_at_start = np.asarray(get_history(alpha_t_start_val))
#     m_current_val = np.asarray(f_dde(t_0, y_current_val, y_delayed_at_start))
#
#     history.append((t_0, y_current_val, m_current_val))
#
#     times_out = [t_0]
#     solutions_out = [y_current_val]
#
#     # --- Main Integration Loop ---
#     current_grid_idx = 0
#
#     # Initialize t_current and y_current from the first point in history_data
#     # These will be the starting state for the first RK4 step.
#     t_current = history[0][0]  # t_start
#     y_current = history[0][1]  # y_initial
#
#     while t_current < t_f:  # Now t_current is properly bound
#         # Determine the next target time (min of h_step or next discontinuity)
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
#             # else: current_grid_idx remains, next_disc_in_grid is not reached this step
#
#         t_next_node = min(t_target_candidate, t_f)  # Final target time for this step
#
#         h_actual = t_next_node - t_current
#         if h_actual < 1e-12:  # Avoid tiny steps or infinite loops if already at t_end
#             break
#
#         # Perform one RK4 step
#         y_next = rk4_step_dde(f_dde, delay, get_history, t_current, y_current, h_actual)
#
#         # Calculate derivative at the new point (m_next) for history storage
#         alpha_t_next_val = delay(t_next_node)
#         y_delayed_at_next = np.asarray(get_history(alpha_t_next_val))
#         m_next = np.asarray(f_dde(t_next_node, y_next, y_delayed_at_next))
#
#         # Add the new (t, y, m) point to history
#
#         history.append((t_next_node, y_next, m_next))
#
#         # Store solution for output
#         times_out.append(t_next_node)
#         solutions_out.append(y_next)
#
#         # Update t_current and y_current for the *next* iteration of the loop
#         t_current = t_next_node
#         y_current = y_next
#
#     # return times_out, solutions_out
#     print(
#         f"history[0]{history[0]} \
#             times_out {times_out[0]} \
#             solutions_out {solutions_out[0]}\
#           "
#     )
#     return times_out, solutions_out, history  # ADDED history_data to the return
