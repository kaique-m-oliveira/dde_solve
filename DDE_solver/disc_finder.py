import numpy as np
from scipy.optimize import root


def find_discontinuity_chain(alpha_func, t_span, h, max_iterations=100):
    """
    Finds a chain of discontinuity points t_0, t_1, t_2, ...
    t_{k+1} is the root of alpha_j(t) - t_k = 0 for some component alpha_j, with t_k < t_{k+1}.
    Accommodates alpha_func(t) returning a scalar or a NumPy array (vector).
    """
    t0, tf = t_span  # Unpack t_span
    discs = [t0]
    tk = t0  # Current known discontinuity point

    # --- Initial check of alpha_func's output to determine vector size ---
    # Use h from signature for a test point guess
    check_t = tk + 10 * h + 10**-3

    try:
        # Get a sample output and force it to be a NumPy array for consistent handling
        test_alpha_output_arr = np.asarray(alpha_func(check_t))
        num_components = (
            test_alpha_output_arr.size
        )  # Number of components to iterate over
        if num_components == 0:
            return [t0]  # No components to process if alpha_func returns empty

    except Exception as e:
        print(
            f"Error during initial alpha_func check at t={check_t}: {e}. Ensure alpha_func returns a number or NumPy array."
        )
        return [t0]

    # --- Main loop to find the chain of discontinuities ---
    for _ in range(max_iterations):
        if tk >= tf:
            break

        candidate_next_roots_this_iter = []

        # Iterate over each component of alpha_func
        for j in range(num_components):
            # Define the equation G(t) = alpha_j(t) - tk = 0 for this specific component
            # 'eq' must return a scalar for scipy.optimize.root
            def eq(tt):  # Using 'tt' as in your template
                alpha_val_for_root = np.asarray(alpha_func(tt))
                return (
                    alpha_val_for_root.flat[j] - tk
                )  # Access component value via flat index

            # Initial guess for the root solver for this component.
            # Using 10 * h as in your template.
            guess = tk + 10 * h
            if guess <= tk:  # Ensure guess is strictly greater than current tk
                guess = tk + 1e-6

            try:
                result = root(eq, guess)

                if result.success:
                    next_t_candidate = result.x[0]  # Extract scalar root

                    # Check constraints: t_k < t_{k+1} and t_{k+1} <= t_end
                    if next_t_candidate > tk + 1e-9 and next_t_candidate <= tf + 1e-9:
                        candidate_next_roots_this_iter.append(next_t_candidate)

            except Exception:
                pass  # Root finding failed for this component/guess, ignore
                # print(f"DEBUG: Root failed for comp {j}, tk={tk}, guess={guess}") # Debugging line

        # If no valid next root was found from any component in this iteration, break the chain
        if not candidate_next_roots_this_iter:
            break

        # Find the smallest valid next root among all components found
        next_tk = np.min(candidate_next_roots_this_iter)

        # Add the found root to the chain and update tk for the next iteration
        discs.append(next_tk)
        tk = next_tk

    return discs


def find_discontinuity_chain_2(alpha_func, t_span, h, max_iterations=100):
    """
    Locates the discontinuity points for non vanishing time delays
    """
    t0, tf = t_span
    discs = [t0]
    tk = t0

    for _ in range(max_iterations):
        if tk >= tf:
            break  # Stop if beyond end time

        # Actual function we have to find the root for
        eq = lambda tt: alpha_func(tt) - tk

        # Initial guess for the next root. Must be > current_tk.
        guess = tk + 10 * h

        try:
            result = root(eq, guess)

            if result.success:
                next_tk = result.x[0]  # Extract scalar root

                # Check constraints: t_k < t_{k+1} and t_{k+1} <= t_end
                if next_tk > tk + 1e-9 and next_tk <= tf:
                    discs.append(next_tk)
                    tk = next_tk
                else:
                    break  # Root found is not in desired sequence or range
            else:
                break  # Root finding failed for this iteration
        except Exception:
            break  # Catch potential errors during root finding

    return discs
