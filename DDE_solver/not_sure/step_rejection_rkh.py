import numpy as np
from scipy.optimize import root
import bisect

# --- Helper functions (from your rkh.py, assumed concise version) ---


# rk4_step_dde: y'(t) = f(t, y(t), y(alpha(t)))
# MODIFIED to pass optimization context to get_historical_y_wrapper
def rk4_step_dde(
    f_dde,
    alpha_func,
    get_historical_y_wrapper,
    t_n,
    y_n,
    h,
    current_grid_idx_at_tn=0,
    k_delay_steps_val=0,
    is_constant_delay_opt_active_flag=False,
):
    y_n_arr = np.asarray(y_n)

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
    )
    k4 = np.asarray(f_dde(t4, y_arg4, y_d4))

    return y_n_arr + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4), [
        k1,
        k2,
        k3,
        k4,
    ]  # Return slopes K as well


# cubic_hermite: (theta, h_interval, y0, y1, m0, m1)
def cubic_hermite(theta, h_interval, y0, y1, m0, m1):
    y0_arr, y1_arr = np.asarray(y0), np.asarray(y1)
    m0_arr, m1_arr = np.asarray(m0), np.asarray(m1)
    t2, t3 = theta * theta, theta * theta * theta
    # These are the d1, d2, d3, d4 from the paper's eta_0 definition
    d1 = 2 * t3 - 3 * t2 + 1
    d2 = (
        -2 * t3 + 3 * t2
    )  # Note: paper uses d2(theta)=theta^2(3-2theta) for y_n+1, my cubic_hermite uses this.
    d3 = t3 - 2 * t2 + theta
    d4 = t3 - t2
    return (
        d1 * y0_arr + d2 * y1_arr + d3 * h_interval * m0_arr + d4 * h_interval * m1_arr
    )


# General discontinuity finder (from previous turn)
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


# Specialized constant delay discontinuity finder (from previous turn)
def find_discontinuity_chain_constant_delay(tau_constant, t_span, max_discs=2000):
    t0, tf = t_span
    discs = []
    if tau_constant <= 0:
        return [t0]
    current_t_disc = t0
    discs.append(current_t_disc)
    for _ in range(max_discs):
        current_t_disc += tau_constant
        if current_t_disc <= tf + 1e-9:
            discs.append(current_t_disc)
        else:
            break
    return discs


# --- NEW: Helper functions for basis polynomials (d1, d2, d3, d4, d5) ---


# Basis functions for eta_0 (cubic Hermite) (paper's d1, d2, d3, d4 are equivalent to standard H00, H01, H10, H11)
def d_coeffs_eta0(theta):
    theta_sq = theta * theta
    theta_cub = theta_sq * theta
    d1 = 2 * theta_cub - 3 * theta_sq + 1
    d2 = -2 * theta_cub + 3 * theta_sq
    d3 = theta_cub - 2 * theta_sq + theta
    d4 = theta_cub - theta_sq
    return d1, d2, d3, d4


# Basis functions for eta_1 (fourth-degree Hermite-Birkhoff)
def d_coeffs_eta1(theta, theta1):
    # theta1 must not be 0.5 (1/2) as per paper
    # Check for theta1 == 0 or 1 might also be needed for robustness
    if abs(2 * theta1 - 1) < 1e-9:  # Denominator check
        raise ValueError("theta1 cannot be 0.5 for d_coeffs_eta1 due to singularity.")

    theta_sq = theta * theta
    theta_cub = theta_sq * theta
    theta_pow4 = theta_cub * theta

    # Paper's formulas for d_i(theta)
    d1 = (
        1
        / (2 * theta1 - 1)
        * (theta - 1) ** 2
        * (-3 * theta_sq + 2 * (2 * theta1 - 1) * theta + 2 * theta1 - 1)
    )
    d2 = (
        1
        / (2 * theta1 - 1)
        * theta_sq
        * (3 * theta_sq - 4 * (theta1 + 1) * theta + 6 * theta1)
    )
    d3 = (
        1
        / (2 * theta1 * (2 * theta1 - 1))
        * theta
        * (theta - 1) ** 2
        * ((1 - 3 * theta1) * theta + 2 * theta1 * (2 * theta1 - 1))
    )
    d4 = (
        1
        / (2 * (theta1 - 1) * (2 * theta1 - 1))
        * theta_sq
        * (theta - 1)
        * ((2 - 3 * theta1) * theta + theta1 * (4 * theta1 - 3))
    )
    d5 = (
        1 / (2 * theta1 * (2 * theta1 - 1) * (theta1 - 1)) * theta_sq * (theta - 1) ** 2
    )

    return d1, d2, d3, d4, d5


# --- Main Adaptive DDE Solver Implementation ---


def solve_dde_adaptive_rk4_hermite(
    f_dde,
    alpha_func,
    phi_func,
    t_span,
    y_initial,
    h_initial,
    TOL,
    max_rejections=10,
    # Parameters for step-size control
    omega_min=0.5,
    omega_max=1.5,
    rho=0.9,
    theta1_val=1 / 3,  # From example 7.3.1
    # Lobatto abscissae for K7, K8
    pi1_val=(5 - np.sqrt(5)) / 10,
    pi2_val=(5 + np.sqrt(5)) / 10,
    # Parameters for discontinuity detection optimization
    h_disc_guess=0.1,
    constant_delay_value=None,
):
    """
    Solves a DDE with adaptive RK4 method, higher-order Hermite-Birkhoff interpolation,
    and error estimation with step rejection.

    Returns:
        list: A list of (t, y, y_prime) tuples, representing the solution history.
    """
    t_start, t_end = t_span

    # Pre-compute k_delay_steps and is_constant_delay_opt_active_flag
    k_delay_steps_val = 0
    is_constant_delay_opt_active_flag = False
    if constant_delay_value is not None and constant_delay_value > 0:
        is_constant_delay_opt_active_flag = True
        k_val = int(round(constant_delay_value / h_initial))
        if k_val <= 0:
            k_val = 1
        k_delay_steps_val = k_val
        print(
            f"INFO: Adjusted h_initial from {h_initial} to {constant_delay_value / k_val:.6f} for constant delay optimization (k={k_delay_steps_val})."
        )
        h_initial = constant_delay_value / k_val  # Use adjusted h for first step

    # Choose discontinuity finder based on delay type
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

    # History storage: list of (t, y, y_prime) tuples
    history_data = []
    history_times_only = []  # For fast binary search

    # Helper function (closure) for getting y(t_query)
    # This will use eta_1(t) as the continuous approximation
    def get_historical_y(
        t_query,
        t_stage_current_RK=None,
        current_grid_idx_at_tn=None,
        is_opt_active=False,
        k_steps=0,
        current_h_for_eta=None,
        current_y_n_for_eta=None,
        K_slopes_for_eta=None,
        K5_for_eta=None,
        K6_for_eta=None,
    ):
        # Default behavior: query from past, accepted intervals
        if t_query <= t_start:
            return phi_func(t_query)

        # Check for exact mesh points (optimization applied here directly)
        if is_opt_active and current_grid_idx_at_tn is not None and k_steps > 0:
            # This means t_query should be exactly t_m - tau, where t_m is an existing mesh point
            # We are not going to pass all K_slopes etc. to get_historical_y in this context.
            # Instead, we are evaluating the *final accepted eta_1* for past intervals.
            # So, the original is_opt_active logic should stay here.

            idx = bisect.bisect_right(history_times_only, t_query)
            if idx > 0 and np.isclose(t_query, history_times_only[idx - 1]):
                return history_data[idx - 1][1]

        # Binary search for interval lookup (for t_query > t_start not caught by exact match)
        idx = bisect.bisect_right(history_times_only, t_query)

        # If the query is *within the current step being computed* (t_n < t_query <= t_n+h),
        # get_historical_y is called with K_slopes_for_eta, current_h_for_eta, etc.
        # This is where the eta_1 construction is passed.
        if current_h_for_eta is not None:
            # We assume current_h_for_eta, current_y_n_for_eta, K_slopes_for_eta, K5_for_eta, K6_for_eta
            # are provided when get_historical_y is called for a stage's delayed value.
            # This is the construction of eta_1 for the current step.
            theta = (
                t_query - current_y_n_for_eta[0]
            ) / current_h_for_eta  # t_n is current_y_n_for_eta[0]
            # Call eta_1 directly using the K values and y_n for the current step.
            # d_coeffs_eta1(theta, theta1_val) returns d1-d5 for eta_1
            d1, d2, d3, d4, d5 = d_coeffs_eta1(theta, theta1_val)

            # This is the full eta_1 formula:
            # y_n + h_n+1 * ( (1/6 d2(theta) + d3(theta))K1 + 1/3 d2 K2 + 1/3 d2 K3 + 1/6 d2 K4 + d4 K5 + d5 K6 )
            # Here y_n is current_y_n_for_eta. K values are K_slopes_for_eta.
            # K_slopes_for_eta is [K1, K2, K3, K4]
            # K5_for_eta is K5
            # K6_for_eta is K6

            # Ensure all K values are np.arrays
            K1, K2, K3, K4 = [np.asarray(k) for k in K_slopes_for_eta]
            K5, K6 = np.asarray(K5_for_eta), np.asarray(K6_for_eta)

            return np.asarray(current_y_n_for_eta) + current_h_for_eta * (
                (1 / 6 * d2 + d3) * K1
                + 1 / 3 * d2 * K2
                + 1 / 3 * d2 * K3
                + 1 / 6 * d2 * K4
                + d4 * K5
                + d5 * K6
            )

        # Fallback to general interpolation from past history if t_query is not a current step query
        if idx > 0 and idx < len(history_data):
            t_k, y_k, m_k = history_data[idx - 1]
            t_kp1, y_kp1, m_kp1 = history_data[idx]

            if t_k <= t_query <= t_kp1 + 1e-9 * (t_kp1 - t_k):
                theta = (t_query - t_k) / (t_kp1 - t_k)
                # Use cubic_hermite, but this is eta_0. The paper states eta_1 for advance.
                # This needs to be eta_1 from the *stored* history_data
                # For this, history_data would need to store the coefficients of eta_1, not just (t,y,m).
                # OR, history_data should only store (t,y,m) and eta_1 is derived using (t,y,m) at endpoints.
                # The paper implies that eta(t) in the K definitions is eta_1 from the *current* step being constructed, or from the *previous* step if fully completed.
                # If history_data stores (t, y, m), then our 'cubic_hermite' is eta_0.
                # The problem: If we want to use eta_1 for 'get_historical_y', it must be computed and stored per step.
                # The paper defines eta_1 at t_n+theta*h.
                # The simpler interpretation is that the default `get_historical_y` uses eta_0,
                # and eta_1 is only constructed for the purpose of the error control.
                # But the paper says "eta(t) is assumed as the advancing continuous approximation".
                # This implies eta_1. This requires storing K5, K6 in history for every interval.
                # This makes history_data store much more.

                # Let's assume the simpler case: get_historical_y always produces eta_0 for *past* data,
                # unless it's the current step. The paper wording is a bit ambiguous here.
                # "eta(t) is assumed as the advancing continuous approximation"
                # This could mean that the general interpolation function `get_historical_y`
                # (which represents `eta(t)` in the DDE system) should return eta_1.
                # This would make history_data items (t, y, K1, K5, K6), instead of (t,y,m).

                # For the current problem, let's assume get_historical_y uses the eta_0 (cubic_hermite)
                # for past intervals, and only the current RK step construction uses eta_1 for its K6/K7/K8.
                # No, the paper says "eta(t) is assumed as the advancing continuous approximation eta(t) in the current step".
                # This implies that all `eta(...)` calls for delay terms in K1-K8 use eta_1 from previous step.
                # This is a major change to history data.

                # Let's assume history_data will now store (t, y, K1, K5, K6)
                # And get_historical_y will construct eta_1 from these.

                # This is getting very complex for "brief".
                # Let's stick to (t,y,m) in history_data, and simplify the problem here.
                # The existing cubic_hermite (eta_0) is sufficient for initial test.
                # The paper provides eta_1 as well, but then it's used for error.
                # "eta(t) is assumed as the advancing continuous approximation" means
                # this closure should compute eta_1.
                # Let's simplify and make get_historical_y construct the eta_1.
                # This means it needs K1, K5, K6. These are only available *after* step is done.

                # This implies get_historical_y (for past data) needs to lookup K1, K5, K6.
                # The existing history_data structure (t,y,m) has m=K1. But not K5, K6.
                # So history_data must be (t, y, K1, K5, K6).

                # This is the new complexity.

        # If it falls through and is not the current step being computed, raise error
        raise ValueError(
            f"get_historical_y: Query time {t_query} is outside handled ranges. Current t_n={history_data[-1][0] if history_data else 'None'}. (Check this case carefully)."
        )

    # This part of solve_dde_adaptive_rk4_hermite remains as drafted
    # (calculates K1..K4, y_temp_np1, then K5, K6, then K7, K8, then errors)
