import numpy as np

A = [[0, 0, 0, 0], [1 / 2, 0, 0, 0], [0, 1 / 2, 0, 0], [0, 0, 1, 0]]
C = [0, 1 / 2, 1 / 2, 1]
B = [1 / 6, 1 / 3, 1 / 3, 1]


def rk4_step_ode(g, t_0, y_0, h):
    y_0 = np.asarray(y_0)
    k1 = np.asarray(g(t_0, y_0))
    k2 = np.asarray(g(t_0 + 0.5 * h, y_0 + 0.5 * h * k1))
    k3 = np.asarray(g(t_0 + 0.5 * h, y_0 + 0.5 * h * k2))
    k4 = np.asarray(g(t_0 + h, y_0 + h * k3))

    y_1 = y_0 + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    # K = [k1, k2, k3, k4]
    return y_1  # , K


def cubic_hermite(theta, g, t0, t1, y0, y1):
    # P(theta) = H00*y0 + H01*y1 + H10*h*m0 + H11*h*m1
    h, m0, m1 = t1 - t0, g(t0, y0), g(t1, y1)  # approx to the slope
    y0_arr, y1_arr = np.asarray(y0), np.asarray(y1)
    m0_arr, m1_arr = np.asarray(m0), np.asarray(m1)

    theta_sq = theta * theta
    theta_cub = theta_sq * theta

    h00 = 2 * theta_cub - 3 * theta_sq + 1
    h10 = theta_cub - 2 * theta_sq + theta
    h01 = -2 * theta_cub + 3 * theta_sq
    h11 = theta_cub - theta_sq

    p_theta = h00 * y0_arr + h01 * y1_arr + h10 * h * m0_arr + h11 * h * m1_arr
    return p_theta
