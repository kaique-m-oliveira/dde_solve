import numpy as np
import numbers

def linear_interpolant(tn, h, c2, yn, y2):

    yn = np.atleast_1d(yn)
    y2 = np.atleast_1d(y2)
    dt2 = c2 * h
    slope = (y2 - yn) / dt2

    def eta(t):
        t_arr = np.atleast_1d(t)
        dt = (t_arr - tn)[:, None]  # shape (m,1)
        out = yn[None, :] + dt * slope[None, :]
        return np.squeeze(out)  # squeeze for scalar t

    def eta_t(t):
        t_arr = np.atleast_1d(t)
        out = np.tile(slope[None, :], (len(t_arr), 1))
        return np.squeeze(out)

    return eta, eta_t


def quadratic_interpolant(tn, h, ci, z0, k1, Yi):
    z0 = np.atleast_1d(z0)
    k1 = np.atleast_1d(k1)
    Yi = np.atleast_1d(Yi)

    dt_target = ci * h
    C = (Yi - z0 - k1 * dt_target) / (dt_target**2)

    def eta(t):
        t_arr = np.atleast_1d(t)
        dt = (t_arr - tn)[:, None]                       # (m,1)
        out = z0[None, :] + k1[None, :] * dt + C[None, :] * (dt**2)
        return np.squeeze(out)

    def eta_t(t):
        t_arr = np.atleast_1d(t)
        dt = (t_arr - tn)[:, None]
        out = k1[None, :] + 2.0 * C[None, :] * dt
        return np.squeeze(out)

    return eta, eta_t

def get_initial_step(problem, solution, Atol, Rtol, order, neutral = False):
    f, alpha = problem.f, problem.alpha
    t0, y0 = solution.t[0], solution.y[0]
    max_step = problem.t_span[-1]
    ndim = problem.ndim
    alpha0 = alpha(t0, y0)
    eta = solution.eta

    Atol = np.full(y0.shape, Atol)
    Rtol = np.full(y0.shape, Rtol)
    scale = Atol + np.abs(y0)*Rtol

    if neutral:
        beta = problem.beta
        beta0 = beta(t0, y0)

    def norm(x):
        return np.linalg.norm(x)/np.sqrt(ndim)

    if not neutral:
        f0 = f(t0, y0, eta(alpha0))
        solution.feval += 1
    else:
        f0 = f(t0, y0, eta(alpha0), eta(beta0))
        solution.feval += 1

    d0 = norm(y0 / scale)
    d1 = norm(f0 / scale)

    if d0 < 1e-5 or d1 < 1e-5:
        h0 = 1e-6
    else:
        h0 = 0.01 * d0 / d1

    y1 = y0 + h0 * f0
    alpha1 = alpha(t0 + h0, y1)
    if np.all(alpha1 <= t0):
        eta_alpha1 = eta(alpha1)
    else:
        eta_alpha1 = y0 + ((alpha1 - t0)/h0)*(y1 - y0)

    if not neutral:
        f1 = f(t0 + h0, y1, eta_alpha1)
        solution.feval += 1
    else:

        beta1 = beta(t0 + h0, y1)
        if np.all(beta1 <= t0):
            eta_beta1 = eta(beta1)
        else:
            eta_beta1 = y0 + ((beta1 - t0)/h0)*(y1 - y0)

        f1 = f(t0 + h0, y1, eta_alpha1, eta_beta1)
        solution.feval += 1

    d2 = norm((f1 - f0) / scale) / h0

    if d1 <= 1e-15 and d2 <= 1e-15:
        h1 = max(1e-6, h0 * 1e-3)
    else:
        h1 = (0.01 / max(d1, d2)) ** (1 / (order + 1))

    return min(100 * h0, h1, max_step, 0.3)



def bisection_method(f, a, b, TOL):
    fa, fb = f(a), f(b)
    while (b - a)/2 > TOL:
        c = (a + b)/2
        fc = f(c)

        if fc == 0:
            return [c - TOL/2, c + TOL/2]

        if fa * fc < 0:
            b = c
            fb = fc

        else:
            a = c
            fa = fc

    return [a, b]



def vectorize_func(func):
    def wrapper(*args, **kwargs):
        return np.atleast_1d(func(*args, **kwargs))
    return wrapper


def validade_arguments(f, alpha, phi, t_span, beta=False, phi_t=False):
    t0, tf = map(float, t_span)
    t_span = [t0, tf]
    y0 = phi(t0)

    if isinstance(y0, numbers.Real) or np.isscalar(y0):
        ndim = 1
    elif isinstance(y0, (list, np.ndarray)):
        ndim = len(y0)
    else:
        raise TypeError(f"Unsupported type for phi(t0): {type(y0)}")

    alpha0 = alpha(t0, y0)
    if isinstance(alpha0, numbers.Real) or np.isscalar(alpha0):
        n_state_delays = 1
    elif isinstance(alpha0, (list, tuple, np.ndarray)):
        n_state_delays = len(alpha0)
    else:
        raise TypeError(f"Unsupported type for alpha(t0, phi(t0)): {alpha0}")

    n_neutral_delays = 0
    if beta:
        beta0 = beta(t0, y0)
        if isinstance(beta0, numbers.Real) or np.isscalar(beta0):
            n_neutral_delays = 1
        elif isinstance(beta0, (list, np.ndarray)):
            n_neutral_delays = len(beta0)
        else:
            raise TypeError(f"Unsupported type for beta(t0, phi(t0)): {beta0}")

    f = vectorize_func(f)
    alpha = vectorize_func(alpha)
    phi = vectorize_func(phi)
    beta = vectorize_func(beta)
    phi_t = vectorize_func(phi_t)
    return ndim, n_state_delays, n_neutral_delays, f, alpha, phi, t_span, beta, phi_t

