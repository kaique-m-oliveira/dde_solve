import numpy as np


# Your concise RK4 step function
def rk4_step_ode(g, t_0, y_0, h):
    y_0_arr = np.asarray(y_0)  # Renamed for clarity within this function
    k1 = np.asarray(g(t_0, y_0_arr))
    k2 = np.asarray(g(t_0 + 0.5 * h, y_0_arr + 0.5 * h * k1))
    k3 = np.asarray(g(t_0 + 0.5 * h, y_0_arr + 0.5 * h * k2))
    k4 = np.asarray(g(t_0 + h, y_0_arr + h * k3))
    return y_0_arr + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


# Your concise cubic Hermite interpolation function
def cubic_hermite(theta, g, t0, t1, y0, y1):
    h_interval = t1 - t0  # Renamed for clarity
    y0_arr, y1_arr = np.asarray(y0), np.asarray(y1)
    m0_arr, m1_arr = np.asarray(g(t0, y0_arr)), np.asarray(g(t1, y1_arr))

    theta_sq = theta * theta
    theta_cub = theta_sq * theta

    h00 = 2 * theta_cub - 3 * theta_sq + 1
    h10 = theta_cub - 2 * theta_sq + theta
    h01 = -2 * theta_cub + 3 * theta_sq
    h11 = theta_cub - theta_sq

    return (
        h00 * y0_arr
        + h01 * y1_arr
        + h10 * h_interval * m0_arr
        + h11 * h_interval * m1_arr
    )


def rk4_solve_basic(g, t_span, y_initial, h_step):
    """
    Solves ODE y'=g(t,y) over t_span using RK4, outputting at discrete steps.
    """
    t_start, t_end = t_span
    t_current = t_start
    y_current = np.asarray(y_initial)

    times_list = [t_current]
    solutions_list = [y_current]

    num_total_steps = int(np.ceil((t_end - t_start) / h_step))

    for _ in range(num_total_steps):
        h_actual = min(h_step, t_end - t_current)
        if h_actual < 1e-12:  # If remaining step is extremely small, break
            break
        y_current = rk4_step_ode(g, t_current, y_current, h_actual)
        t_current += h_actual
        times_list.append(t_current)
        solutions_list.append(y_current)
    return times_list, solutions_list
