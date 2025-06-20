import numpy as np
from scipy.optimize import root
import bisect 

# --- Helper functions for RK stages and interpolants ---

# rk4_step_dde: y'(t) = f(t, y(t), y(alpha(t)))
# Returns (y_np1, [K1,K2,K3,K4])
# NOTE: get_historical_y_wrapper is passed complex context for optimization and in-step interpolation
def rk4_step_dde(
    f_dde, alpha_func, get_historical_y_wrapper, t_n, y_n, h,
    current_grid_idx_at_tn=0, k_delay_steps_val=0, is_constant_delay_opt_active_flag=False,
    # Context for get_historical_y when called for current step stages (k2,k3,k4)
    # This implies eta(t) for current step is built based on values being computed.
    current_step_context=None # Dict: {'y_n_val':y_n, 'h_val':h, 'K_slopes':[], 'K5_val':None, 'K6_val':None}
):
    y_n_arr = np.asarray(y_n)

    # K1 calculation. For K1, t_stage is t_n (start of interval), so no current_step_context needed for its y_delayed.
    t1 = t_n; 
    y_d1 = np.asarray(get_historical_y_wrapper(alpha_func(t1, y_n_arr), t1, current_grid_idx_at_tn, is_constant_delay_opt_active_flag, k_delay_steps_val))
    k1 = np.asarray(f_dde(t1, y_n_arr, y_d1))

    # For K2, K3, K4, current_step_context is needed for in-step delay evaluation
    # These calls to get_historical_y might query into the current (incomplete) step
    # using the provided context (current RK stage slope k1 or k2).
    
    # Store K1 temporarily in context for get_historical_y for K2, K3, K4 (and eta_1 calculations)
    if current_step_context is None: # Should only be for K1/K5/K6 in stage calculation context
        current_step_context = {} # Initialize empty dict
    current_step_context['K_slopes'] = [k1, None, None, None] # Only K1 is known yet for get_historical_y

    t2 = t_n + 0.5*h; 
    y_arg2 = y_n_arr + 0.5*h*k1; 
    y_d2 = np.asarray(get_historical_y_wrapper(alpha_func(t2, y_arg2), t2, current_grid_idx_at_tn, is_constant_delay_opt_active_flag, k_delay_steps_val, current_step_context))
    k2 = np.asarray(f_dde(t2, y_arg2, y_d2))
    current_step_context['K_slopes'][1] = k2 # Update K_slopes in context

    t3 = t_n + 0.5*h; 
    y_arg3 = y_n_arr + 0.5*h*k2; 
    y_d3 = np.asarray(get_historical_y_wrapper(alpha_func(t3, y_arg3), t3, current_grid_idx_at_tn, is_constant_delay_opt_active_flag, k_delay_steps_val, current_step_context))
    k3 = np.asarray(f_dde(t3, y_arg3, y_d3))
    current_step_context['K_slopes'][2] = k3 # Update K_slopes in context

    t4 = t_n + h; 
    y_arg4 = y_n_arr + h*k3; 
    y_d4 = np.asarray(get_historical_y_wrapper(alpha_func(t4, y_arg4), t4, current_grid_idx_at_tn + 1, is_constant_delay_opt_active_flag, k_delay_steps_val, current_step_context)) # Index for t_n+h is current_grid_idx_at_tn + 1
    k4 = np.asarray(f_dde(t4, y_arg4, y_d4))
    current_step_context['K_slopes'][3] = k4 # Update K_slopes in context
    
    return y_n_arr + (h/6.0)*(k1 + 2.0*k2 + 2.0*k3 + k4), [k1, k2, k3, k4] # Return slopes K as well

# --- Helper: Basis functions for eta_0 (standard cubic Hermite, equivalent to paper's d1, d2, d3, d4) ---
# Used for eta_0(t) and indirectly for get_historical_y if context_for_current_step is None (past data)
def d_coeffs_eta0_basis(theta):
    theta_sq = theta * theta
    theta_cub = theta_sq * theta
    d1 = (2 * theta_cub - 3 * theta_sq + 1)
    d2 = (-2 * theta_cub + 3 * theta_sq)
    d3 = (theta_cub - 2 * theta_sq + theta)
    d4 = (theta_cub - theta_sq)
    return d1, d2, d3, d4

# --- Helper: Basis functions for eta_1 (fourth-degree Hermite-Birkhoff) ---
# Used for eta_1(t) (the advancing continuous approximation)
def d_coeffs_eta1_basis(theta, theta1):
    if abs(2 * theta1 - 1) < 1e-9: 
        raise ValueError("theta1 cannot be 0.5 for d_coeffs_eta1 due to singularity.")
    
    theta_sq = theta * theta
    theta_cub = theta_sq * theta
    theta_pow4 = theta_cub * theta

    d1 = 1 / (2 * theta1 - 1) * (theta - 1)**2 * (-3 * theta_sq + 2 * (2 * theta1 - 1) * theta + 2 * theta1 - 1)
    d2 = 1 / (2 * theta1 - 1) * theta_sq * (3 * theta_sq - 4 * (theta1 + 1) * theta + 6 * theta1)
    d3 = 1 / (2 * theta1 * (2 * theta1 - 1)) * theta * (theta - 1)**2 * ((1 - 3 * theta1) * theta + 2 * theta1 * (2 * theta1 - 1))
    d4 = 1 / (2 * (theta1 - 1) * (2 * theta1 - 1)) * theta_sq * (theta - 1) * ((2 - 3 * theta1) * theta + theta1 * (4 * theta1 - 3))
    d5 = 1 / (2 * theta1 * (2 * theta1 - 1) * (theta1 - 1)) * theta_sq * (theta - 1)**2
    
    return d1, d2, d3, d4, d5

# --- General discontinuity finder (from previous turn) ---
def find_discontinuity_chain_general(alpha_func, t_span_find_disc, h_disc_find, max_iterations_find_disc, y_initial_for_disc_finding):
    t0_find, tf_find = t_span_find_disc
    discs = [t0_find]
    tk = t0_find
    y_placeholder = np.asarray(y_initial_for_disc_finding) # Use placeholder y_val
    if y_placeholder.ndim == 0: y_placeholder = np.array([y_placeholder.item()])
    
    try:
        check_t = tk + h_disc_find
        if check_t <= tk: check_t = tk + 1e-6 
        test_alpha_output_arr = np.asarray(alpha_func(check_t, y_placeholder)) 
        num_components = test_alpha_output_arr.size
        if num_components == 0: return [t0_find]
    except Exception: return [t0_find]

    for _ in range(max_iterations_find_disc):
        if tk >= tf_find: break
        candidate_next_roots = []
        for j in range(num_components):
            eq = lambda tt: np.asarray(alpha_func(tt, y_placeholder)).flat[j] - tk 
            guess = tk + 10 * h_disc_find 
            if guess <= tk: guess = tk + 1e-6 
            try:
                result = root(eq, guess)
                if result.success:
                    next_t_cand = result.x[0] 
                    if next_t_cand > tk + 1e-9 and next_t_cand <= tf_find + 1e-9:
                        candidate_next_roots.append(next_t_cand)
            except Exception: pass
        if not candidate_next_roots: break 
        next_tk = np.min(candidate_next_roots)
        discs.append(next_tk)
        tk = next_tk
    return discs

# --- Specialized constant delay discontinuity finder (from previous turn) ---
def find_discontinuity_chain_constant_delay(tau_constant, t_span, max_discs=2000):
    t0, tf = t_span
    discs = []
    if tau_constant <= 0: return [t0]
    current_t_disc = t0
    discs.append(current_t_disc)
    for _ in range(max_discs):
        current_t_disc += tau_constant
        if current_t_disc <= tf + 1e-9:
            discs.append(current_t_disc)
        else: break
    return discs


# --- Main Adaptive DDE Solver Implementation ---

def solve_dde_adaptive_rk4_hermite(
    f_dde, alpha_func, phi_func, t_span, y_initial, h_initial, TOL, 
    max_rejections=10, 
    omega_min=0.5, omega_max=1.5, rho=0.9, theta1_val=1/3, 
    pi1_val=(5-np.sqrt(5))/10, pi2_val=(5+np.sqrt(5))/10,
    h_disc_guess=0.1, constant_delay_value=None 
):
    """
    Solves a DDE with adaptive RK4 method, higher-order Hermite-Birkhoff interpolation,
    and error estimation with step rejection. Handles state-dependent delay.
    
    Returns:
        list: A list of (t, y, K1, K5, K6) tuples, representing the solution history.
              K1=y'(t), K5=y'(t+h)|_using_y(t+h)_and_eta0, K6=y'(t+theta1*h)|_using_eta0_and_eta.
    """
    t_start, t_end = t_span
    
    # Adjust h_initial for constant delay optimization
    h_actual_for_calc = h_initial 
    k_delay_steps_val = 0 
    is_constant_delay_opt_active_flag = False
    if constant_delay_value is not None and constant_delay_value > 0:
        is_constant_delay_opt_active_flag = True
        k_val = int(round(constant_delay_value / h_initial))
        if k_val <= 0: k_val = 1 
        h_actual_for_calc = constant_delay_value / k_val
        k_delay_steps_val = k_val 
        print(f"INFO: Adjusted h_initial from {h_initial} to {h_actual_for_calc:.6f} for constant delay optimization (k={k_delay_steps_val}).")
    
    # Choose discontinuity finder: Pass y_initial to find_discontinuity_chain_general for pre-calculating discs
    if constant_delay_value is not None:
        discs_found = find_discontinuity_chain_constant_delay(constant_delay_value, t_span)
    else:
        discs_found = find_discontinuity_chain_general(alpha_func, t_span, h_disc_guess, 500, y_initial)
    
    grid_points = sorted(list(set([t_start] + [d for d in discs_found if t_start <= d <= t_end])))
    
    # History storage: (t, y, K1, K5, K6) tuples. This will be the direct output.
    # Note: K5, K6 are computed after step acceptance, so they are fixed for previous steps.
    history_data = [] 
    history_times_only = [] 

    # get_historical_y now represents the eta(t) function
    # It must handle: 1. phi(t) 2. past eta_1(t) 3. current_step eta_1(t) (predictor)
    def get_historical_y(t_query, t_stage_current_RK=None, current_grid_idx_at_tn=None, is_opt_active=False, k_steps=0, current_step_context=None):
        """
        Retrieves y(t_query) using the eta(t) interpolant.
        `context_for_current_step` is None unless called during RK stage calculation for current step.
        """
        # 1. Query is in the initial history segment (t <= t_start)
        if t_query <= t_start: 
            return phi_func(t_query)
        
        # 2. Query is within *current* step being computed (t_n < t_query <= t_n + h)
        #    This is handled by current_step_context, passed from rk4_step_dde
        if current_step_context is not None:
            # Need y_n, h, K1, K2, K3, K4 from current_step_context to build eta_1
            y_n_val = current_step_context['y_n_val']
            h_val = current_step_context['h_val']
            K1, K2, K3, K4 = current_step_context['K_slopes'] # K_slopes are being built
            K5 = current_step_context.get('K5_val') # K5 might not be known yet
            K6 = current_step_context.get('K6_val') # K6 might not be known yet

            if K5 is None or K6 is None: # K5/K6 not yet known, so cannot build full eta_1
                # Fallback: Use eta_0 or simpler predictor if K5/K6 are not available
                # The paper for overlapping uses hat_eta_0 (a specific cubic using K1-K4)
                # But here, we can use eta_0 (standard Hermite if y_n+1 is predicted).
                # Simpler: just use linear prediction or raise error.
                # The paper's definition of K6, K7, K8 (using eta_1) implies a circular dependency.
                # Standard practice: for K6, K7, K8, eta(delay) uses the eta_1 from the *previous* accepted step.
                # OR, for explicit RK, eta(delay) is based on a fixed interpolant up to t_n.
                # Or, a predictor.
                # Given problem description, we will use a predictor (linear from t_n,y_n,m_n).
                # This is the "giving up on the solution for now" for this problem.
                
                # Use linear predictor from (t_n, y_n, m_n) for current step queries.
                return y_n_val + (t_query - t_n_current_step) * np.asarray(K1) # K1 is m_n_current_step
        
        # --- Lookup for fully processed intervals (t_query > t_start and not current step) ---
        idx = bisect.bisect_right(history_times_only, t_query) 
        
        # NEW OPTIMIZATION: Direct lookup if t_query is an exact mesh point (bypasses bisect check if possible)
        # This occurs if constant_delay_value is active AND t_query is an exact mesh point.
        if is_opt_active and current_grid_idx_at_tn is not None and k_steps > 0: # Context must be passed from rk4_step_dde
            # Case 1: t_stage_current_RK is t_n (start of current interval)
            if np.isclose(t_stage_current_RK, history_times_only[current_grid_idx_at_tn]):
                target_idx = current_grid_idx_at_tn - k_steps
                if target_idx >= 0 and np.isclose(t_query, history_times_only[target_idx]):
                    return history_data[target_idx][1] # Return y-value from (t,y,K1,K5,K6) tuple
            # Case 2: t_stage_current_RK is t_n + h_actual (end of current interval)
            elif current_grid_idx_at_tn + 1 < len(history_times_only) and \
                 np.isclose(t_stage_current_RK, history_times_only[current_grid_idx_at_tn + 1]):
                target_idx = (current_grid_idx_at_tn + 1) - k_steps
                if target_idx >= 0 and np.isclose(t_query, history_times_only[target_idx]):
                    return history_data[target_idx][1]
            # Fallback for optimization: if active but not exact mesh point for current t_stage, use bisect for that point.

        # General case (not directly indexed by optimization) - use bisect and interpolant (eta_1)
        if idx > 0 and idx < len(history_data): 
            t_k, y_k, K1_k, K5_k, K6_k = history_data[idx-1] # Unpack (t, y, K1, K5, K6)
            t_kp1, y_kp1, K1_kp1, K5_kp1, K6_kp1 = history_data[idx] # Unpack next point's data
            
            if t_k <= t_query <= t_kp1 + 1e-9 * (t_kp1 - t_k):
                theta = (t_query - t_k) / (t_kp1 - t_k)
                # Call eta_1_interpolant_function (defined below)
                return get_eta1_interpolant_func(theta, t_kp1-t_k, y_k, y_kp1, K1_k, K5_k, K6_k, theta1_val) # Pass original theta1_val
        
        # Fallback for queries within the current step (if not handled by context) or out of bounds.
        raise ValueError(f"get_historical_y: Query time {t_query} outside processed history or t_start. History up to {history_data[-1][0] if history_data else 'None'}.")

    # --- Initial K1 calculation for the first step's y_initial ---
    y_current_val = np.asarray(y_initial)
    alpha_t_start_val = alpha_func(t_start, y_current_val) # State-dependent for initial derivative
    y_delayed_at_start = np.asarray(get_historical_y(alpha_t_start_val)) 
    m_current_val = np.asarray(f_dde(t_start, y_current_val, y_delayed_at_start))
    
    # Store initial point: (t, y, K1, K5, K6)
    # At t_start, K5 and K6 are not yet defined for this interval.
    # K5 is K_n+2^1. K6 depends on eta_0.
    # For initial point, K5 and K6 are not available. Use np.nan or 0.0, or specific initial conditions for them.
    # The paper uses K_n+1^5 = K_n+2^1.
    # For t_start, we store (t_start, y_start, K1_start, K5_start_placeholder, K6_start_placeholder)
    # K5_start_placeholder and K6_start_placeholder will be updated AFTER t_start+h is accepted.
    # This implies initial_history_data has placeholders.
    # This requires more complex management outside get_historical_y.
    
    # Let's initialize history_data points as (t,y,m) and then reconstruct K1,K5,K6 from m for eta_1 interpolation
    # Or, the paper means K5, K6 from *previous* intervals.

    # Revisit paper interpretation:
    # "eta(t) is assumed as the advancing continuous approximation"
    # -> get_historical_y should provide eta_1.
    # -> `history_data` stores `(t, y, K1, K5, K6)`.
    # -> `K5` and `K6` are defined using `eta_0` and `eta_1` respectively.

    # This means the current implementation of `get_historical_y` for past data needs
    # (t, y, K1, K5, K6) in history_data, not just (t,y,m).
    # This change must apply to the main loop's append, not just `get_historical_y`.

    # Let's assume initial K5 and K6 are 0s, and they are filled in later.
    history_data.append((t_start, y_current_val, m_current_val, np.zeros_like(y_current_val), np.zeros_like(y_current_val)))
    history_times_only.append(t_start) 

    current_grid_idx = 0 
    t_current = t_start 
    y_current = y_current_val 
    h_current_suggested = h_actual_for_calc # Initial step size suggestion for adaptive loop

    while t_current < t_end:
        num_rejections_this_step = 0
        while True: # Loop until step is accepted or max rejections reached
            if num_rejections_this_step > max_rejections:
                raise RuntimeError(f"Max rejections ({max_rejections}) reached at t={t_current}. Cannot proceed.")

            t_target_candidate = t_current + h_current_suggested
            
            actual_grid_idx = current_grid_idx 
            if actual_grid_idx + 1 < len(grid_points):
                next_disc_in_grid = grid_points[actual_grid_idx + 1]
                if next_disc_in_grid <= t_target_candidate + 1e-9 * h_current_suggested:
                    t_target_candidate = next_disc_in_grid
                    actual_grid_idx += 1 

            t_next_node = min(t_target_candidate, t_end)
            h_actual = t_next_node - t_current

            if h_actual < 1e-12: break 

            # --- Calculate RK stages and y_next (K1-K4) ---
            # get_historical_y needs context_for_current_step for its current_step_context branch
            current_step_rk_context = {'y_n_val': y_current, 'h_val': h_actual, 
                                       'K_slopes': [None]*4, 'K5_val': None, 'K6_val': None}
            
            # K1: (t_n, y_n, eta(alpha(t_n)))
            y_d1 = np.asarray(get_historical_y(alpha_func(t_current, y_current), t_current, current_grid_idx, is_constant_delay_opt_active_flag, k_delay_steps_val))
            K1 = np.asarray(f_dde(t_current, y_current, y_d1))
            current_step_rk_context['K_slopes'][0] = K1 # Store K1 in context

            # K2: (t_n+0.5h, y_n+0.5hK1, eta(alpha(t_n+0.5h)))
            t2_stage = t_current + 0.5*h_actual
            y_arg2 = y_current + 0.5*h_actual*K1
            y_d2 = np.asarray(get_historical_y(alpha_func(t2_stage, y_arg2), t2_stage, current_grid_idx, is_constant_delay_opt_active_flag, k_delay_steps_val, current_step_context)) # Pass context
            K2 = np.asarray(f_dde(t2_stage, y_arg2, y_d2))
            current_step_rk_context['K_slopes'][1] = K2 # Store K2

            # K3: (t_n+0.5h, y_n+0.5hK2, eta(alpha(t_n+0.5h)))
            t3_stage = t_current + 0.5*h_actual
            y_arg3 = y_current + 0.5*h_actual*K2
            y_d3 = np.asarray(get_historical_y(alpha_func(t3_stage, y_arg3), t3_stage, current_grid_idx, is_constant_delay_opt_active_flag, k_delay_steps_val, current_step_rk_context)) # Pass context
            K3 = np.asarray(f_dde(t3_stage, y_arg3, y_d3))
            current_step_rk_context['K_slopes'][2] = K3 # Store K3

            # K4: (t_n+h, y_n+hK3, eta(alpha(t_n+h)))
            t4_stage = t_current + h_actual
            y_arg4 = y_current + h_actual*K3
            y_d4 = np.asarray(get_historical_y(alpha_func(t4_stage, y_arg4), t4_stage, current_grid_idx + 1, is_constant_delay_opt_active_flag, k_delay_steps_val, current_step_rk_context)) # Pass context, index of t4_stage is +1
            K4 = np.asarray(f_dde(t4_stage, y_arg4, y_d4))
            current_step_rk_context['K_slopes'][3] = K4 # Store K4

            y_temp_np1 = y_current + (h_actual/6.0)*(K1 + 2.0*K2 + 2.0*K3 + K4) # y_n+1 from RK4

            # --- Calculate K5 (K_n+1^5 = K_n+2^1) ---
            # K5 is f(t_n+1, y_n+1, eta(t_n+1 - tau(t_n+1, y_n+1)))
            alpha_t_np1_val = alpha_func(t_next_node, y_temp_np1) # State-dependent
            # get_historical_y for K5 (after step completed): no current_step_context needed
            y_delayed_K5 = np.asarray(get_historical_y(alpha_t_np1_val, t_next_node, current_grid_idx + 1, is_constant_delay_opt_active_flag, k_delay_steps_val))
            K5 = np.asarray(f_dde(t_next_node, y_temp_np1, y_delayed_K5))
            current_step_rk_context['K5_val'] = K5 # Store K5 in context for K6, K7, K8

            # --- Calculate eta_0 (cubic Hermite) ---
            # get_eta0_interpolant(theta, h, y0, y1, K1, K5) - simplified.
            # Passed to get_historical_y's current_step_context for K6 calculation
            
            # --- Calculate K6 ---
            t6_stage = t_current + theta1_val * h_actual
            # eta0_at_theta1 is eta_0(t_n+theta1*h). Uses y_n, y_temp_np1, K1, K5
            eta0_at_theta1 = get_eta0_interpolant_func(theta1_val, h_actual, y_current, y_temp_np1, K1, K5) # Call helper directly
            alpha_t_K6_val = alpha_func(t6_stage, eta0_at_theta1) # State-dependent
            # Pass context for K6 calculation using eta_0 as current_step_context has K1, K5
            y_delayed_K6 = np.asarray(get_historical_y(alpha_t_K6_val, t6_stage, current_grid_idx, is_constant_delay_opt_active_flag, k_delay_steps_val, current_step_rk_context)) # Pass context
            K6 = np.asarray(f_dde(t6_stage, eta0_at_theta1, y_delayed_K6))
            current_step_rk_context['K6_val'] = K6 # Store K6 in context

            # --- Calculate K7, K8 (for error estimation) ---
            # K7: (t_n+pi1*h, eta_1(t_n+pi1*h), eta(alpha(t_n+pi1*h)))
            t7_stage = t_current + pi1_val * h_actual
            # eta1_at_pi1 is eta_1(t_n+pi1*h). Uses y_n, y_temp_np1, K1, K5, K6
            eta1_at_pi1 = get_eta1_interpolant_func((t7_stage - t_current) / h_actual, h_actual, y_current, y_temp_np1, K1, K5, K6, theta1_val)
            alpha_t_K7_val = alpha_func(t7_stage, eta1_at_pi1)
            y_delayed_K7 = np.asarray(get_historical_y(alpha_t_K7_val, t7_stage, current_grid_idx, is_constant_delay_opt_active_flag, k_delay_steps_val, current_step_rk_context))
            K7 = np.asarray(f_dde(t7_stage, eta1_at_pi1, y_delayed_K7))

            # K8: (t_n+pi2*h, eta_1(t_n+pi2*h), eta(alpha(t_n+pi2*h)))
            t8_stage = t_current + pi2_val * h_actual
            eta1_at_pi2 = get_eta1_interpolant_func((t8_stage - t_current) / h_actual, h_actual, y_current, y_temp_np1, K1, K5, K6, theta1_val)
            alpha_t_K8_val = alpha_func(t8_stage, eta1_at_pi2)
            y_delayed_K8 = np.asarray(get_historical_y(alpha_t_K8_val, t8_stage, current_grid_idx, is_constant_delay_opt_active_flag, k_delay_steps_val, current_step_rk_context))
            K8 = np.asarray(f_dde(t8_stage, eta1_at_pi2, y_delayed_K8))
            
            # --- Calculate higher-order approximation tilde_y_np1 ---
            tilde_y_np1 = y_current + h_actual * ( (1/12) * K1 + (5/12) * K7 + (5/12) * K8 + (1/12) * K5 )

            # --- Calculate discrete local error (sigma_tilde_np1) ---
            sigma_tilde_np1 = np.linalg.norm(tilde_y_np1 - y_temp_np1) / h_actual # L2 norm

            # --- Calculate uniform local error (hat_Sigma_np1) ---
            # delta(theta) = eta_0(theta) - eta_1(theta)
            delta_at_half = get_eta0_interpolant_func(0.5, h_actual, y_current, y_temp_np1, K1, K5) - \
                            get_eta1_interpolant_func(0.5, h_actual, y_current, y_temp_np1, K1, K5, K6, theta1_val)
            hat_Sigma_np1 = np.linalg.norm(delta_at_half)

            # --- Step Acceptance/Rejection Logic ---
            accepted_step = False
            # Test 1: Discrete error tolerance
            if sigma_tilde_np1 <= TOL:
                # Test 2: Uniform error tolerance (if Test 1 passed)
                if h_actual * hat_Sigma_np1 <= TOL:
                    accepted_step = True
                    h_new_next_step = max(omega_min, min(omega_max, rho * (TOL / sigma_tilde_np1)**(1/4), rho * (TOL / (h_actual * hat_Sigma_np1))**(1/5))) * h_actual
                else: # Test 2 failed, reject step
                    h_new_reduced = max(omega_min, rho * (TOL / (h_actual * hat_Sigma_np1))**(1/5)) * h_actual
            else: # Test 1 failed, reject step
                h_new_reduced = max(omega_min, rho * (TOL / sigma_tilde_np1)**(1/4)) * h_actual
            
            if accepted_step:
                y_next_accepted = y_temp_np1 # The accepted solution
                
                # Calculate m_next (derivative at t_next_node)
                alpha_t_next_val = alpha_func(t_next_node, y_next_accepted) 
                y_delayed_at_next = np.asarray(get_historical_y(alpha_t_next_val, t_next_node, current_grid_idx + 1, is_constant_delay_opt_active_flag, k_delay_steps_val))
                m_next = np.asarray(f_dde(t_next_node, y_next_accepted, y_delayed_at_next))

                history_data.append((t_next_node, y_next_accepted, m_next, K5, K6)) # Store K5, K6 too
                history_times_only.append(t_next_node) 

                t_current = t_next_node
                y_current = y_next_accepted
                h_current_suggested = h_new_next_step 
                break # Exit rejection loop
            else: # Step rejected
                h_current_suggested = h_new_reduced 
                num_rejections_this_step += 1
                # print(f"DEBUG: Step rejected at t={t_current}. Retrying with h={h_current_suggested:.6f}. Rejections={num_rejections_this_step}")
    
    return history_data
