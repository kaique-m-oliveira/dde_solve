import time
import numbers
from bisect import bisect_left
from dataclasses import dataclass, field
import random
import matplotlib.pyplot as plt
import numpy as np
# from scipy.optimize import root
from scipy.integrate import solve_ivp


# def real_sol(t):
#     if t <= 0:
#         return -t
#     if 0 <= t <= 1:
#         return -2 + t + 2*np.exp(t)
#     elif 1 <= t <= 2:
#         return 4 - t + 2*np.exp(t) - 2*(t + 1)*np.exp(t - 1)
#
#
# def real_sol_t(t):
#     if t <= 0:
#         return -1
#     if 0 <= t <= 1:
#         return 1 + 2*np.exp(t)
#     if 1 <= t <= 2:
#         return -1 + 2*np.exp(t) - 2*(t+2)*np.exp(t - 1)


def real_sol(t):
    return np.log(t)


def real_sol_t(t):
    return 1/t


def lu_factor(A):
    """LU decomposition with partial pivoting, LAPACK-style.
    Returns LU, piv such that P@A = L@U,
    where LU stores L (unit diag, strictly lower part) and U (upper part).
    """
    A = A.copy().astype(float)
    m = A.shape[0]
    piv = np.arange(m)

    for k in range(m - 1):
        # pivot selection
        idx = k + np.argmax(np.abs(A[k:m, k]))
        if A[idx, k] == 0:
            raise ValueError("Matrix is singular.")

        # swap rows in A
        if idx != k:
            A[[k, idx], :] = A[[idx, k], :]
            piv[[k, idx]] = piv[[idx, k]]

        # elimination
        for j in range(k + 1, m):
            A[j, k] /= A[k, k]
            A[j, k+1:m] -= A[j, k] * A[k, k+1:m]

    return A, piv


def lu_solve(lu_and_piv, b):
    """Solve Ax=b given LU decomposition from lu_factor.
    Args:
        lu_and_piv: (LU, piv) from lu_factor
        b: right-hand side (vector or matrix)
    Returns:
        x: solution of Ax=b
    """
    LU, piv = lu_and_piv
    m = LU.shape[0]
    b = np.array(b, dtype=float, copy=True)

    b = b[piv]

    # Forward substitution (solve L y = Pb)
    for i in range(m):
        b[i] -= np.dot(LU[i, :i], b[:i])

    # Back substitution (solve U x = y)
    for i in reversed(range(m)):
        b[i] -= np.dot(LU[i, i+1:], b[i+1:])
        b[i] /= LU[i, i]

    return b


def root(dz, t_guess, t_span, method='hybr', tol=np.finfo(float).eps, max_iter=50):
    """
    Simple bisection root finder for scalar dz(t) with a known sign change in [tn, tn+h].
    Drop-in replacement for root(..., method='hybr').
    """
    a, b = t_span
    fa, fb = dz(a), dz(b)

    # Expand interval to [tn, tn+h] if needed
    while fa * fb > 0:
        a -= 1e-3
        b += 1e-3
        fa, fb = dz(a), dz(b)
        if a < 0 or b > t_guess + 2.0:
            # fail-safe
            return type("RootResult", (), {"x": t_guess, "success": False})()

    for _ in range(max_iter):
        c = 0.5 * (a + b)
        fc = dz(c)
        if abs(fc) < tol:
            return type("RootResult", (), {"x": c, "success": True})()
        if fa * fc < 0:
            b, fb = c, fc
        else:
            a, fa = c, fc

    # last estimate
    return type("RootResult", (), {"x": 0.5*(a+b), "success": abs(dz(0.5*(a+b))) < tol})()


@dataclass
class CRKParameters:
    theta1: float = 1 / 3
    TOL: float = 1e-5
    rho: float = 0.9
    omega_min: float = 0.2 #was 0.5
    omega_max: float = 1.5 #between 1.5 and 5


def vectorize_func(func):
    def wrapper(*args, **kwargs):
        # return np.array(func(*args, **kwargs))
        return np.atleast_1d(func(*args, **kwargs))
    return wrapper


def validade_arguments(f, alpha, phi, t_span, d_f=None, d_alpha=None, d_phi=None, beta=None, neutral=None):
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
    elif isinstance(alpha0, (list, np.ndarray)):
        n_state_delays = len(alpha0)
    else:
        raise TypeError(f"Unsupported type for alpha(t0, phi(t0)): {alpha0}")

    n_neutral_delays = 0
    if beta is not None:
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
    return ndim, n_state_delays, n_neutral_delays, f, alpha, phi, t_span, beta


class RungeKutta:
    def __init__(self, problem, solution, h, neutral=False, Atol = 1e-7, Rtol = 1e-7):

        A: np.ndarray = NotImplemented
        b: np.ndarray = NotImplemented
        b_err: np.ndarray = NotImplemented
        c: np.ndarray = NotImplemented
        D: np.ndarray = NotImplemented
        D_err: np.ndarray = NotImplemented
        D_ovl: np.ndarray = NotImplemented
        order: dict = NotImplemented
        n_stages: dict = NotImplemented

        total_stages = self.A.shape[0]
        tn = solution.t[-1]
        yn = solution.y[-1]
        self.problem = problem
        self.solution = solution
        self.h = h
        self.t = [tn, tn + self.h]
        self.h_next = None
        self.y = [yn, 1]  # we don't have yn_plus yet
        self.n = problem.ndim
        self.y_tilde = None
        self.K = np.zeros((total_stages, self.n), dtype=float)
        self.eta = solution.eta
        self.eta_t = solution.eta_t
        self.new_eta = [None, None]
        self.new_eta_t = [None, None]
        self.new_eta_tt = None
        self.disc_local_error = None
        self.uni_local_error = None
        self.params = CRKParameters()
        self.overlap = False
        self.test = False
        self.disc = False
        self.ndim = problem.ndim
        self.ndelays = problem.n_state_delays
        self.fails = 0
        self.stages_calculated = 0
        self.store_times = []
        self.number_of_calls = 0
        self.neutral = neutral
        self.Atol = np.full(self.y[0].shape, Atol)
        self.Rtol = np.full(self.y[0].shape, Rtol)

    @property
    def eeta(self):
        def eval(t):
            t = np.atleast_1d(t)  # accept scalar or array

            results = np.empty((len(t), self.problem.ndim), dtype=float)
            for i in range(len(t)):
                if t[i] <= self.t[0]:
                    results[i] = self.solution.eta(t[i])
                else:
                    if self.new_eta[1] is not None:
                        results[i] = self.new_eta[1](t[i])
                    elif self.new_eta[0] is not None:
                        results[i] = self.new_eta[0](t[i])
                    else:
                        results[i] = self._hat_eta_0(t[i])
            return np.squeeze(results)
        return eval

    @property
    def eeta_t(self):
        def eval(t):
            t = np.atleast_1d(t)  # accept scalar or array
            results = np.empty((len(t), self.problem.ndim), dtype=float)
            for i in range(len(t)):
                if t[i] <= self.t[0]:
                    results[i] = self.solution.eta_t(t[i])

                else:
                    if self.new_eta[1] is not None:
                        results[i] = self.new_eta_t[1](t[i])
                    elif self.new_eta[0] is not None:
                        results[i] = self.new_eta_t[0](t[i])
                    else:
                        results[i] = self._hat_eta_0_t(t[i])
                # else:
                #     results[i] = self._hat_eta_0_t(t[i])
            return np.squeeze(results)
        return eval

    @property
    def eeta_tt(self):
        def eval(t):
            t = np.atleast_1d(t)  # accept scalar or array
            results = np.empty((len(t), self.problem.ndim), dtype=float)
            for i in range(len(t)):
                if t[i] <= self.t[0]:
                    results[i] = self.solution.eta_tt(t[i])
                else:
                    results[i] = self._hat_eta_0_tt(t[i])
            return np.squeeze(results)
        return eval

    def is_there_disc(self):
        tn, h = self.t[0], self.h
        eta, alpha = self.solution.etas[-1], self.problem.alpha
        d_alpha = self.problem.d_alpha
        if self.neutral:
            beta = self.problem.beta
        discs = self.solution.discs
        hn = self.solution.t[-1] - self.solution.t[-2]

        if hn <= 1e-15:
            return False

        theta = 1 + h/hn

        def d_zeta(delay, t, disc):
            return delay(t, eta(t)) - disc  # np.full(self.ndelays, disc)


        for disc in discs:

            sign_change_alpha = d_zeta(alpha, tn, disc) * d_zeta(alpha, tn + theta * h, disc) < 0
            new_disc = None
            if np.any(sign_change_alpha):
                new_disc = self.get_disc(alpha, d_alpha, disc, sign_change_alpha)

            if self.neutral:
                sign_change_beta = d_zeta(beta, tn, disc) * d_zeta(beta, tn + theta * h, disc) < 0
                if np.any(sign_change_beta):
                    d_beta = self.problem.d_beta
                    new_disc_beta = self.get_disc(beta, d_beta, disc, sign_change_beta)
                    if new_disc is not None:
                        new_disc = min(new_disc, new_disc_beta)

            if new_disc is not None:
                self.disc = new_disc
                return True
        return False

    def get_disc(self, delay, d_delay, disc, disc_position):
        delay_t, delay_y = d_delay
        eta = self.solution.etas[-1]
        t_guess = self.t[0] + self.h/2
        indices = np.where(disc_position)[0].tolist()

        t_roots = []

        for idx in indices:

            def d_zeta(t):
                return delay(t, eta(t))[idx] - disc

            # print('d_zeta(t_guess)', d_zeta(t_guess), 'shape', d_zeta(t_guess))
            sol = root(d_zeta, t_guess, [self.t[0], self.t[0] + self.h], method='hybr')
            # sol1 = root(d_zeta, t_guess, method='hybr')
            # print('sol.x', sol.x)
            # print('sol.x', sol1.x)
            t_roots.append(sol.x)

        return min(t_roots)

    def one_step_RK4(self):
        # print('SSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSS')
        # print('SSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSS')
        tn, h, yn = self.t[0], self.h, self.y[0]

        f, eta, alpha = self.problem.f, self.eta, self.problem.alpha
        n_stages = self.n_stages["discrete_method"]
        c = self.c[:n_stages]
        A = self.A[:n_stages, :n_stages]
        if self.neutral:
            self.K[0] = f(tn, yn, eta(alpha(tn, yn)),
                          self.eta_t(self.problem.beta(tn, yn)))
        else:
            self.K[0] = f(tn, yn, eta(alpha(tn, yn)))
        self.stages_calculated = 1

        for i in range(1, n_stages):
            ti = tn + c[i] * h
            yi = yn + h * (A[i][0:i] @ self.K[0: i])

            if np.all(alpha(ti, yi) <= np.full(self.ndelays, tn)):
                alpha_i = alpha(ti, yi)
                Y_tilde = eta(alpha_i)
                # print('eta(alpha_i)', Y_tilde,
                #       'real_sol(alpha_i)', real_sol(alpha_i), 'diff', abs(Y_tilde - real_sol(alpha_i)))
                if self.neutral:
                    beta_i = self.problem.beta(ti, yi)
                    Z_tilde = self.eta_t(alpha_i)
                    # print('---eta_t(alpha_i)', Z_tilde,
                    #       'real_sol_t(alpha_i)', real_sol_t(alpha_i), 'diff', abs(Z_tilde - real_sol_t(alpha_i)))
                    self.K[i] = f(ti, yi, Y_tilde, Z_tilde)
                else:
                    self.K[i] = f(ti, yi, Y_tilde)
                self.stages_calculated = i + 1
            else:  # this would be the overlapping case
                self.overlap = True
                success = self._simplified_Newton(alpha(ti, yi))
                if not success:
                    return False
                break

        self.y[1] = yn + h * (self.b @ self.K[0:n_stages])
        self.stages_calculated = n_stages

        # print(f'tn = {tn}, h = {
        #       h}, yn+1 = {self.y[1]}, yn.shape {self.y[1].shape} ')
        # # print('FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF')
        # print('FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF')
        # input('RK4 stuff')
        return True

    def _simplified_Newton(self, alpha_i):
        # print('[[[[[[[[[[[[[[[[[[[[[[[[[ BEGINNING NEWTON ]]]]]]]]]]]]]]]]]]]]]]]]]')
        # print('t', self.t[0])

        first_stage = self.stages_calculated
        final_stage = self.n_stages["continuous_ovl_method"]
        total_stages = final_stage - first_stage if final_stage > first_stage else 1

        A = self.A[first_stage:final_stage, first_stage:final_stage]
        c = self.c
        rho, TOL = self.params.rho, self.params.TOL

        alpha_t, alpha_y = self.problem.d_alpha

        if self.neutral:
            f_t, f_y, f_x, f_z = self.problem.d_f
            beta_t, beta_y = self.problem.d_beta
        else:
            f_t, f_y, f_x = self.problem.d_f

        eta_t = self.solution.eta_t
        eta_tt = self.solution.eta_tt

        tn, h, yn = self.t[0], self.h, self.y[0]
        f, eta = self.problem.f, self.eta
        alpha, beta = self.problem.alpha, self.problem.beta

        alpha_n = alpha(tn, yn)
        alpha_y_n = alpha_y(tn, yn)

        if self.neutral:
            beta_n = beta(tn, yn)
            f_y_n = f_y(tn, yn, eta(alpha_n), eta_t(beta_n))
            f_x_n = f_x(tn, yn, eta(alpha_n), eta_t(beta_n))
            f_z_n = f_z(tn, yn, eta(alpha_n), eta_t(beta_n))
            beta_y_n = beta_y(tn, yn)

        else:
            f_y_n = f_y(tn, yn, eta(alpha_n))
            f_x_n = f_x(tn, yn, eta(alpha_n))

        pol_order = self.D_ovl.shape[1]

        #FIX: collapsing into scalar, WRONG
        sum_1 = np.sum(f_x_n * self.eeta_t(alpha_n) * alpha_y_n)

        sum_1t = 0
        if self.neutral:
            #FIX: collapsing into scalar, WRONG
            sum_1t = np.sum(f_z_n * self.eeta_tt(beta_n) * beta_y_n)

        theta = np.squeeze((alpha_i - tn) / h)

        sum_2 = 0
        for i in range(self.ndelays):
            if alpha_i[i] <= self.t[-1]:
                B = np.zeros((total_stages, total_stages), dtype=yn.dtype)

            else:
                theta_i = theta[i] ** np.arange(pol_order)
                D_ovl = self.D_ovl[first_stage:final_stage, :]
                D = D_ovl @ theta_i
                B = np.tile(D[:, None], (1, total_stages))

            kron = np.kron(B, f_x_n[i])

            # WARN: HAVEN'T DECIDED WHETHER TO USE THETA_I OR THETA_N
            kron_t = 0
            if self.neutral:
                # print('neutral here')
                if beta_n[i] <= self.t[-1]:
                    B_t = np.zeros(
                        (total_stages, total_stages), dtype=yn.dtype)
                else:
                    theta_i = np.array([n*theta**(n-1)
                                     for n in range(pol_order)])
                    D_ovl = self.D_ovl[first_stage:final_stage, :]
                    D = D_ovl @ theta_i
                    B_t = np.tile(D[:, None], (1, total_stages))
                kron_t = np.kron(B_t, f_z_n[i])

            sum_2 += kron + kron_t

        I = np.kron(np.eye(total_stages, dtype=yn.dtype),
                    np.eye(self.ndim, dtype=yn.dtype))

        first = - h * np.kron(A, f_y_n + sum_1 + sum_1t)
        second = - h * sum_2
        J = I + first + second

        lu, piv = lu_factor(J)

        def F(K):
            F = np.zeros((total_stages, self.ndim), dtype=float)
            for i in range(total_stages):
                ti = tn + c[first_stage + i] * h
                yi = yn + c[first_stage + i] * h * K[i]

                Y_tilde = self.eeta(alpha(ti, yi))
                if self.neutral:
                    Z_tilde = self.eeta_t(beta(ti, yi))
                    # print('_____________F______________')
                    # print('beta_i', beta(ti, yi))
                    # print('z', Z_tilde, 'real_t', real_sol_t(beta(ti, yi)))
                    # print('f in z', f(ti, yi, Y_tilde, Z_tilde))
                    F[i] = K[i] - f(ti, yi, Y_tilde, Z_tilde)
                else:
                    F[i] = K[i] - f(ti, yi, Y_tilde)
            return F.ravel()

        max_iter, iter = 130, 0
        err = 100

        # first iteration
        first_guess = np.full(self.K[first_stage:final_stage].shape, self.K[0])
        # print(self.K[first_stage:final_stage].shape)
        self.K[first_stage:final_stage] = first_guess
        # print('stages to calculate:', total_stages)
        # print('K0', self.K[0])
        # print('self.K', self.K)
        # print('first_guess', first_guess)
        FK = F(first_guess)

        diff_old = lu_solve((lu, piv), - FK).reshape(total_stages, self.ndim)
        self.K[first_stage:final_stage] += diff_old
        # print('K', self.K[first_stage:final_stage])
        iter += 1

        inside_K = self.K[first_stage:final_stage]
        FK = F(inside_K)
        diff_new = lu_solve((lu, piv), - FK).reshape(total_stages, self.ndim)
        self.K[first_stage:final_stage] += diff_new
        # print('K', self.K[first_stage:final_stage])
        iter += 1
        err = abs((np.linalg.norm(diff_new)**2) /
                  (np.linalg.norm(diff_old) - np.linalg.norm(diff_new)))

        # while err >= rho * TOL and iter <= max_iter:
        while np.any(F(inside_K) >= rho * TOL) and iter <= max_iter:

            # Método de Newton usando recomposição LU
            diff_old = diff_new

            inside_K = self.K[first_stage:final_stage]
            FK = F(inside_K)
            diff_new = lu_solve(
                (lu, piv), - FK).reshape(total_stages, self.ndim)
            self.K[first_stage:final_stage] += diff_new
            # print('K', self.K[first_stage:final_stage])
            iter += 1
            err = abs((np.linalg.norm(diff_new)**2) /
                      (np.linalg.norm(diff_old) - np.linalg.norm(diff_new)))
            # print('err', err)
            # print('err2', np.max(F(inside_K)))
            # input('stop')

        # print('Ks', self.K)
        # print('final K', self.K[first_stage:final_stage])
        # input('[[[[[[[[[[[[[[[[[[[[[[[[[ ENDING NEWTON ]]]]]]]]]]]]]]]]]]]]]]]]]')
        if iter > max_iter:

            # print('err', err)
            # print('TOL', TOL, 'rho', rho, rho*TOL)
            # y1 = yn + h * (self.b @ self.K[0:4])
            # print(f'tn = {tn}, h = {
            #       h}, yn+1 = {y1}, yn.shape {y1.shape} ')
            # print(f'yn+1 = {y1} real_sol = {real_sol(tn + h)
            #                                        } diff={y1 - real_sol(tn + h)}')
            # input(f'{'A'*400} ')
            print('Newton falhou')
            return False
        # input(f'deu certo com erro {err} e {iter} iterações')
        return True

    def build_eta_0(self):
        f, alpha = self.problem.f,  self.problem.alpha
        if self.n_stages["continuous_err_est_method"] - self.stages_calculated <= 0:
            return
        else:
            for i in range(self.stages_calculated, self.n_stages["continuous_err_est_method"]):
                ti = self.t[0] + self.c[i] * self.h
                yi = self.y[0] + self.h * (self.A[i][0:i] @ self.K[0: i])
                alpha_i = alpha(ti, yi)
                Y_tilde = self.eeta(alpha_i)
                if self.neutral:
                    beta_i = self.problem.beta(ti, yi)
                    Z_tilde = self.eeta_t(alpha_i)
                    self.K[i] = f(ti, yi, Y_tilde, Z_tilde)
                else:
                    self.K[i] = f(ti, yi, Y_tilde)
            self.stages_calculated = self.n_stages["continuous_err_est_method"]

    def _eta_0(self, theta):
        tn, h, yn = self.t[0], self.h, self.y[0]
        theta = (theta - tn) / h
        if h<=1e-13:
            print('h', h)
            input('yeah')

        pol_order = self.D_err.shape[1]
        theta = theta ** np.arange(pol_order)
        K = self.K[0:self.n_stages["continuous_err_est_method"]]
        bs = (self.D_err @ theta).squeeze()
        eta0 = yn + h * bs @ K

        return eta0


    def _eta_0_t(self, theta):
        tn, h, yn = self.t[0], self.h, self.y[0]
        theta = (theta - tn) / h
        pol_order = self.D_err.shape[1]
        theta = np.array([n*theta**(n-1) for n in range(pol_order)])
        K = self.K[0:self.n_stages["continuous_err_est_method"]]
        bs = (self.D @ theta).squeeze()
        eta0 = bs @ K
        return eta0

    def _hat_eta_0(self, theta):

        tn, h, yn = self.t[0], self.h, self.y[0]
        theta = (theta - tn) / h
        pol_order = self.D_ovl.shape[1]
        theta = theta ** np.arange(pol_order)
        K = self.K[0:self.n_stages["continuous_ovl_method"]]
        bs = (self.D_ovl @ theta).squeeze()
        eta0 = yn + h * bs @ K

        return eta0

    def _hat_eta_0_t(self, theta):
        # print('------------------inside hat_t ------------------------')
        realval_t = real_sol_t(theta)
        tn, h, yn = self.t[0], self.h, self.y[0]
        theta = (theta - tn) / h
        pol_order = self.D_ovl.shape[1]
        theta = np.array([n*theta**(n-1) for n in range(pol_order)])
        K = self.K[0:self.n_stages["continuous_ovl_method"]]
        # print('K', K)
        # print('theta', theta)
        bs = (self.D_ovl @ theta).squeeze()
        # print('D@theta', self.D_ovl@theta)
        # print('bs', bs)
        # eta0 = yn + h * bs @ K
        eta0 = bs @ K
        # print('hat0', eta0, 'real', realval_t, 'diff', abs(realval_t - eta0))
        # print('---------------------------------------------------------')
        return eta0

    def _hat_eta_0_tt(self, theta):
        tn, h, yn = self.t[0], self.h, self.y[0]
        theta = (theta - tn) / h
        pol_order = self.D_ovl.shape[1]
        theta = np.array([n*(n-1)*theta**(n-2) for n in range(pol_order)])
        K = self.K[0:self.n_stages["continuous_ovl_method"]]
        bs = (self.D_ovl @ theta).squeeze()
        # eta0 = yn + h * bs @ K
        eta0 = bs @ K
        return eta0

    def build_eta_1(self):
        f, alpha = self.problem.f,  self.problem.alpha
        if self.n_stages["continuous_method"] - self.stages_calculated <= 0:
            return
        else:
            for i in range(self.stages_calculated, self.n_stages["continuous_method"]):
                ti = self.t[0] + self.c[i] * self.h
                yi = self.y[0] + self.h * (self.A[i][0:i] @ self.K[0: i])
                alpha_i = alpha(ti, yi)
                Y_tilde = self.eeta(alpha_i)
                if self.neutral:
                    beta_i = self.problem.beta(ti, yi)
                    Z_tilde = self.eeta_t(alpha_i)
                    self.K[i] = f(ti, yi, Y_tilde, Z_tilde)
                else:
                    self.K[i] = f(ti, yi, Y_tilde)
            self.stages_calculated = self.n_stages["continuous_method"]

    def _eta_1(self, theta):
        tn, h, yn = self.t[0], self.h, self.y[0]
        theta = (theta - tn) / h

        pol_order = self.D.shape[1]
        theta = theta ** np.arange(pol_order)
        K = self.K[0:self.n_stages["continuous_method"]]
        bs = (self.D @ theta).squeeze()
        eta0 = yn + h * bs @ K
        return eta0

    def _eta_1_t(self, theta):
        tn, h, yn = self.t[0], self.h, self.y[0]
        theta = (theta - tn) / h
        pol_order = self.D.shape[1]
        theta = np.array([n*theta**(n-1) for n in range(pol_order)])
        K = self.K[0:self.n_stages["continuous_method"]]
        bs = (self.D @ theta).squeeze()
        eta0 = bs @ K
        return eta0

    def _eta_1_tt(self, theta):
        tn, h, yn = self.t[0], self.h, self.y[0]
        theta = (theta - tn) / h
        pol_order = self.D.shape[1]
        theta = np.array([n*(n-1)*theta**(n-2) for n in range(pol_order)])
        K = self.K[0:self.n_stages["continuous_method"]]
        bs = (self.D @ theta).squeeze()
        eta0 = bs @ K
        return eta0

    def error_est_method(self):
        f, alpha = self.problem.f,  self.problem.alpha
        if self.n_stages["discrete_err_est_method"] - self.stages_calculated <= 0:
            K = self.K[0:self.n_stages["discrete_err_est_method"]]
            self.y_tilde = self.y[0] + self.h * (self.b_err @ K)
            return
        else:
            for i in range(self.stages_calculated, self.n_stages["discrete_err_est_method"]):
                ti = self.t[0] + self.c[i] * self.h
                yi = self.y[0] + self.h * (self.A[i][0:i] @ self.K[0: i])
                alpha_i = alpha(ti, yi)
                Y_tilde = self.eeta(alpha_i)
                if self.neutral:
                    beta_i = self.problem.beta(ti, yi)
                    Z_tilde = self.eeta_t(alpha_i)
                    self.K[i] = f(ti, yi, Y_tilde, Z_tilde)
                else:
                    self.K[i] = f(ti, yi, Y_tilde)

            self.stages_calculated = self.n_stages["discrete_err_est_method"]

        K = self.K[0:self.n_stages["discrete_err_est_method"]]
        self.y_tilde = self.y[0] + self.h * (self.b_err @ K)

    def disc_local_error_satistied(self):
        sc = self.Atol +  np.maximum(np.abs(self.y[1]), np.abs(self.y_tilde))*self.Rtol
        self.disc_local_error = (
            np.linalg.norm((self.y_tilde - self.y[1])/sc)/np.sqrt(self.ndim)
        )  # eq 7.3.4
        err2 = np.linalg.norm(self.y_tilde - self.y[1])  # eq 7.3.4

        # print('_______________________disc__________________________')
        # print('sc', sc)
        # print('disc err1', self.disc_local_error, 'disc err2', err2, 'diff', abs(self.disc_local_error - err2))



        # if self.disc_local_error <= self.params.TOL:
        if self.disc_local_error <= 1:
            return True
        else:
            # self.h = min(1, self. h * (
            #     max(
            #         self.params.omega_min,
            #         min(
            #             self.params.omega_max,
            #             self.params.rho
            #             * (self.params.TOL / self.disc_local_error) ** (1 / 4),
            #         ),
            #     )
            # ))
            facmax = self.params.omega_max
            facmin = self.params.omega_min
            fac = self.params.rho
            err1 = self.disc_local_error
            pp = 4 #FIX: adicionar o agnóstico
            s = (1/err1)**(1/(pp + 1))
            new_h = self.h* min(facmax, max(facmin, fac*s ))
            # input(f'failed new h = {new_h} old h = {self.h}')
            self.h = new_h

            self.t = [self.t[0], self.t[0] + self.h]
            return False

    def uni_local_error_satistied(self):
        # print('_______________________uni__________________________')

        tn, h = self.t[0], self.h
        val1 = self.new_eta[0](tn + h/2)
        val2 = self.new_eta[1](tn + h/2)

        err2 = h * np.linalg.norm(val1 - val2) #eq 7.3.4

        # print('err2', err2)
        # print('val1', val1, 'val2', val2)
        # print('max', np.maximum(val1, val2))
        # print('max*Rtol', np.maximum(val1, val2)*self.Rtol)
        # print('self.Atol', self.Atol)
        # print('self.Rtol', self.Rtol)
        sc = self.Atol +  np.maximum(np.abs(val1), np.abs(val2))*self.Rtol
        # print('sc', sc)

        self.uni_local_error = (
            np.linalg.norm((val1 - val2)/sc)/np.sqrt(self.ndim)
        )  # eq 7.3.4
        # print('err1', self.uni_local_error, 'err2', err2, 'diff', abs(self.uni_local_error - err2))

        # if self.uni_local_error <= self.params.TOL:
        if self.uni_local_error <= 1:
            return True
        else:
            # self.h = min(1, self.h * (
            #     max(
            #         self.params.omega_min, self.params.rho *
            #         (self.params.TOL / self.uni_local_error) ** (1 / 5)
            #     )
            # ))

            facmax = self.params.omega_max
            facmin = self.params.omega_min
            fac = self.params.rho
            err2 = self.uni_local_error
            qq = 4 #FIX: adicionar o agnóstico
            s = (1/err2)**(1/(qq + 1))
            self.h = self.h* min(facmax, max(facmin, fac*s))

            self.t = [self.t[0], self.t[0] + self.h]
            return False

    def try_step_CRK(self):
        success = self.one_step_RK4()
        if not success:
            self.h = self.h/2
            self.h_next = self.h
            return False

        self.build_eta_1()
        self.new_eta[1] = self._eta_1
        self.build_eta_0()
        self.new_eta[0] = self._eta_0
        self.new_eta_t = [self._eta_0_t, self._eta_1_t]
        self.new_eta_tt = self._eta_1_tt
        self.error_est_method()

        # print('\n\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n\n')
        local_disc_satisfied = self.disc_local_error_satistied()

        uni_local_disc_satistied = self.uni_local_error_satistied()

        # input(f'stop at t = [{self.t[0]}, {self.t[0] + self.h}] h = {self.h}')


        if not local_disc_satisfied:
            print(f'failed disc t = {
                self.t[0] + self.h}, h = {self.h}, err = {self.disc_local_error}')
            # print('y1', self.y[1], 'y1*', self.y_tilde, 'diff', abs(self.y[1] - self.y_tilde))
            return False

        if not uni_local_disc_satistied:
            print(f'failed uni t = {
                  self.t[0] + self.h}, h = {self.h} err = {self.uni_local_error} ')
            return False

        # print(f'successfull step with h = {self.h}')
        # Handling divide by zero case
        if self.disc_local_error < 1e-14 or self.uni_local_error < 1e-14:
            self.h_next = min(1, self.params.omega_max * self.h)
        else:
            # self.h_next = min(1, self.h * max(
            #     self.params.omega_min,
            #     min(
            #         self.params.omega_max,
            #         self.params.rho * (self.params.TOL /
            #                            self.disc_local_error)**(1/4),
            #         self.params.rho * (self.params.TOL /
            #                            self.uni_local_error)**(1/5)
            #     )
            # ))

            facmax = self.params.omega_max
            facmin = self.params.omega_min
            fac = self.params.rho
            err1 = self.disc_local_error
            err2 = self.uni_local_error
            pp = 4 #FIX: adicionar o agnóstico
            qq = 3
            self.h_next = self.h* min(facmax, max(facmin, (1/err1)**(1/pp + 1))) #, (1/err2)**(1/qq + 1)))

        return True

    def one_step_CRK(self, max_iter=100):
        time1 = time.time()
        calls1 = self.solution.eta_calls
        success = self.try_step_CRK()
        time2 = time.time()
        calls2 = self.solution.eta_calls
        # print('calls', calls2 - calls1)
        # print('times', time2 - time1)
        if success:
            return True, self
        else:
            for i in range(max_iter - 1):
                time1 = time.time()
                calls1 = self.solution.eta_calls
                disc_found = self.is_there_disc()
                if disc_found:
                    new_h = self.disc - self.t[0]
                    # print('new_h', new_h, 'type', type(new_h))
                    # print('self.h', self.h)
                    # print('self.h_next ', self.h_next)
                    if new_h < self.h and new_h > 10**-12:
                        print('disc_found', disc_found)
                        if self.disc not in self.solution.discs:
                            # self.solution.discs.append(disc_found)
                            self.solution.discs.append(self.disc)
                            # print('self.solution.discs', self.solution.discs)
                            # input(f'mais uma disc {self.disc}')
                        self.h = float(new_h)
                        self.t = [self.t[0], self.t[0] + self.h]
                    # input('stop')
                success = self.try_step_CRK()
                calls2 = self.solution.eta_calls
                time2 = time.time()
                # print('calls', calls2 - calls1)
                # print('time', time2 - time1)
                if success:
                    return True, self

            return False, 0

    def first_step_CRK(self, max_iter=100):
        self.eta = self.solution.eta
        # print('why')
        success = self.try_step_CRK()
        if success:
            return True, self
        else:
            for i in range(max_iter - 1):
                success = self.try_step_CRK()
                if success:
                    return True, self

            return False, 0


class RK4HHL(RungeKutta):
    A = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0],
        [1/2, 0, 0, 0, 0, 0, 0, 0],
        [0, 1/2, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0],
        [1/6, 1/3, 1/3, 1/6, 0, 0, 0, 0],
        [31/162, 7/81, 7/81, 7/162, -2/27, 0, 0, 0],
        [(37 - np.sqrt(5))/300, (7 * (1 - np.sqrt(5)))/150, (7 * (1 - np.sqrt(5))) /
         150, (7 * (1 - np.sqrt(5)))/300, (-1 + 2 * np.sqrt(5))/100, 27/100, 0, 0],
        [(np.sqrt(5) + 37)/300, (7 * (1 + np.sqrt(5)))/150, (7 * (1 + np.sqrt(5))) /
         150, (7 * (1 + np.sqrt(5)))/300, -(1 + 2 * np.sqrt(5))/100, 27/100, 0, 0]
    ], dtype=np.float64)

    b_err = np.array(
        [1/12, 0, 0, 0, 1/12, 0, 5/12, 5/12], dtype=np.float64)

    b = np.array([1/6, 1/3, 1/3, 1/6], dtype=np.float64)

    c = np.array([0, 1/2, 1/2, 1, 1, 1/3, (5 - np.sqrt(5)) /
                  10, (5 + np.sqrt(5))/10], dtype=np.float64)

    D = np.array([
        [0, 1, -3, 11/3, -3/2],
        [0, 0, -2, 16/3, -3],
        [0, 0, -2, 16/3, -3],
        [0, 0, -1, 8/3, -3/2],
        [0, 0, 5/4, -7/2, 9/4],
        [0, 0, 27/4, -27/2, 27/4]
    ], dtype=np.float64)

    D_err = np.array([
        [0, 1, -3/2, 2/3],
        [0, 0, 1, -2/3],
        [0, 0, 1, -2/3],
        [0, 0, 1/2, -1/3],
        [0, 0, -1, 1]
    ], dtype=np.float64)

    D_ovl = np.array([
        [0, 1, -3/2, 2/3],
        [0, 0, 1, -2/3],
        [0, 0, 1, -2/3],
        [0, 0, -1/2, 2/3]
    ], dtype=np.float64)

    order = {"discrete_method": 4, "discrete_err_est_method": 5,
             "continuous_method": 4, "continuous_err_est_method": 3, "continuous_ovl_method": 3}

    n_stages = {"discrete_method": 4, "discrete_err_est_method": 8,
                "continuous_method": 6, "continuous_err_est_method": 5, "continuous_ovl_method": 4}


class RK45(RungeKutta):
    # Dormand–Prince RK5(4) with continuous extension

    A = np.array([
        [0, 0, 0, 0, 0, 0, 0],
        [1/5, 0, 0, 0, 0, 0, 0],
        [3/40, 9/40, 0, 0, 0, 0, 0],
        [44/45, -56/15, 32/9, 0, 0, 0, 0],
        [19372/6561, -25360/2187, 64448/6561, -212/729, 0, 0, 0],
        [9017/3168, -355/33, 46732/5247, 49/176, -5103/18656, 0, 0],
        [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0]
    ], dtype=np.float64)

    # 5th-order weights
    b = np.array([35/384, 0, 500/1113, 125/192,
                  -2187/6784, 11/84], dtype=np.float64)

    # Error estimate weights (difference to 4th-order)
    b_err = np.array([5179/57600, 0, 7571/16695, 393/640,
                      -92097/339200, 187/2100, 1/40], dtype=np.float64)

    # Stage time fractions
    c = np.array([0, 1/5, 3/10, 4/5, 8/9, 1, 1], dtype=np.float64)

    # Dense output coefficients (like your D matrix)
    D = np.array([
        [ 0.        ,  1.        , -2.86053867,  3.09957788, -1.16181058,  0.01391721],
        [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,  0.        ],
        [ 0.        ,  0.        ,  4.04714150, -6.34535405,  2.79546509, -0.04801624],
        [ 0.        ,  0.        , -3.94116007, 10.90400303, -6.72931751,  0.41751622],
        [ 0.        ,  0.        ,  2.84194470, -7.54767586,  4.95763672, -0.57428174],
        [ 0.        ,  0.        , -1.61098864,  4.21891584, -2.95010387,  0.47312904],
        [ 0.        ,  0.        ,  1.52360117, -4.32946684,  3.08813015, -0.28226449]
    ], dtype=np.float64)

    D_ovl = D
    D_err = D

    order = {
        "discrete_method": 5,
        "discrete_err_est_method": 4,
        "continuous_method": 4,
        "continuous_err_est_method": 4,
        "continuous_ovl_method": 4
    }


    n_stages = {
        "discrete_method": 6,
        "discrete_err_est_method": 7,   # b_err has 7 entries
        "continuous_method": 7,   # dense output uses 7 coeffs
        "continuous_err_est_method": 7, 
        "continuous_ovl_method": 7
    }





class Problem:
    def __init__(self, f, alpha, phi, t_span, d_f=None, d_alpha=None, d_phi=None, beta=None, neutral=False):
        ndim, n_state_delays, n_neutral_delays, f, alpha, phi, t_span, beta = validade_arguments(
            f, alpha, phi, t_span, d_f = d_f, d_alpha = d_alpha, d_phi = d_phi, beta = beta, neutral = neutral)
        self.t_span = np.array(t_span)
        self.ndim, self.n_state_delays, self.n_neutral_delays, self.f, self.alpha, self.phi, self.t_span = ndim, n_state_delays, n_neutral_delays, f, alpha, phi, t_span
        self.beta = beta
        self.y_type = np.zeros(self.ndim, dtype=float).dtype
        self.d_alpha = self.get_delay_t(alpha)
        self.d_phi_t = None

        if beta is not None:
            self.d_beta = self.get_delay_t(beta)

        if d_phi is not None:
            self.d_phi = d_phi
        else:
            self.d_phi = self.get_d_phi(phi)

        if neutral:
            self.d_f = self.get_d_f_neutral()
            self.d_phi_t = self.get_d_phi(self.d_phi)
        else:
            self.d_f = self.get_d_f()

    def get_d_phi(self, phi):
        h = 1e-15

        def d_phi(t):
            return (phi(t) - phi(t - h))/h

        return d_phi

    def get_d_f(self):
        alpha = self.alpha
        f = self.f
        ndim = self.ndim
        ndelays = self.n_state_delays
        d_alpha = np.empty(ndelays, dtype=self.y_type)
        h = 1e-15

        def unit_vec(j): return np.array(
            [1 if i == j else 0 for i in range(ndim)])

        def f_t(t, y, x):
            return (f(t, y, x) - f(t - h, y, x))/h

        def f_y(t, y, x):
            val = np.zeros((self.ndim, self.ndim), dtype=float)
            for j in range(ndim):
                val_j = (f(t, y, x) - f(t, y - h*unit_vec(j), x))/h
                val[j] = val_j
            # input('f_y')
            return np.atleast_1d(val)

        def x_add(x, h, j):
            x[j] -= h
            return x

        if ndelays == 1:
            def f_x(t, y, x):
                # delays = np.empty((ndelays, ndelays * ndim), dtype=y.dtype)
                delays = np.zeros((ndelays, ndim, ndim), dtype=y.dtype)
                for i in range(ndelays):
                    # val = np.zeros(self.ndim, dtype=float)
                    val = np.zeros((self.ndim, self.ndim), dtype=float)
                    for j in range(ndim):
                        val[j] = (f(t, y, x) - f(t, y, x - h*unit_vec(j)))/h
                    delays[i] = np.atleast_1d(val)
                # return np.squeeze(delays)
                return delays

        else:
            def f_x(t, y, x):
                # delays = np.empty(ndelays, dtype=y.dtype)
                # delays = np.empty((ndelays, ndim), dtype=y.dtype)
                delays = np.zeros((ndelays, ndim, ndim), dtype=y.dtype)
                for i in range(ndelays):
                    # val = np.zeros(self.ndim, dtype=float)
                    val = np.zeros((self.ndim, self.ndim), dtype=float)
                    for j in range(ndim):
                        val[j] = (f(t, y, x) - f(t, y, x_add(x, h, j)))/h
                    delays[i] = val
                # return np.squeeze(delays)
                return delays

        return f_t, f_y, f_x

    def get_d_f_neutral(self):
        alpha = self.alpha
        f = self.f
        ndim = self.ndim
        ndelays = self.n_state_delays
        d_alpha = np.empty(ndelays, dtype=self.y_type)
        h = 1e-15

        def unit_vec(j): return np.array(
            [1 if i == j else 0 for i in range(ndim)])

        def f_t(t, y, x, z):
            return (f(t, y, x, z) - f(t - h, y, x, z))/h

        def f_y(t, y, x, z):
            val = np.zeros((self.ndim, self.ndim), dtype=float)
            for j in range(ndim):
                val_j = (f(t, y, x, z) - f(t, y - h*unit_vec(j), x, z))/h
                val[j] = val_j
            # input('f_y')
            return np.atleast_1d(val)

        def x_add(x, h, j):
            x[j] -= h
            return x

        if ndelays == 1:
            def f_x(t, y, x, z):
                # delays = np.empty((ndelays, ndelays * ndim), dtype=y.dtype)
                delays = np.zeros((ndelays, ndim, ndim), dtype=y.dtype)
                for i in range(ndelays):
                    # val = np.zeros(self.ndim, dtype=float)
                    val = np.zeros((self.ndim, self.ndim), dtype=float)
                    for j in range(ndim):
                        val[j] = (f(t, y, x, z) -
                                  f(t, y, x - h*unit_vec(j), z))/h
                    delays[i] = np.atleast_1d(val)
                # return np.squeeze(delays)
                return delays

            def f_z(t, y, x, z):
                # delays = np.empty((ndelays, ndelays * ndim), dtype=y.dtype)
                delays = np.zeros((ndelays, ndim, ndim), dtype=y.dtype)
                for i in range(ndelays):
                    # val = np.zeros(self.ndim, dtype=float)
                    val = np.zeros((self.ndim, self.ndim), dtype=float)
                    for j in range(ndim):
                        val[j] = (f(t, y, x, z) -
                                  f(t, y, x, z - h*unit_vec(j)))/h
                    delays[i] = np.atleast_1d(val)
                # return np.squeeze(delays)
                return delays

        else:
            def f_x(t, y, x, z):
                # delays = np.empty(ndelays, dtype=y.dtype)
                # delays = np.empty((ndelays, ndim), dtype=y.dtype)
                delays = np.zeros((ndelays, ndim, ndim), dtype=y.dtype)
                for i in range(ndelays):
                    # val = np.zeros(self.ndim, dtype=float)
                    val = np.zeros((self.ndim, self.ndim), dtype=float)
                    for j in range(ndim):
                        val[j] = (f(t, y, x, z) - f(t, y, x_add(x, h, j), z))/h
                    delays[i] = val
                # return np.squeeze(delays)
                return delays

            def f_z(t, y, x, z):
                # delays = np.empty(ndelays, dtype=y.dtype)
                # delays = np.empty((ndelays, ndim), dtype=y.dtype)
                delays = np.zeros((ndelays, ndim, ndim), dtype=y.dtype)
                for i in range(ndelays):
                    # val = np.zeros(self.ndim, dtype=float)
                    val = np.zeros((self.ndim, self.ndim), dtype=float)
                    for j in range(ndim):
                        val[j] = (f(t, y, x, z) - f(t, y, x, x_add(z, h, j)))/h
                    delays[i] = val
                # return np.squeeze(delays)
                return delays

        return f_t, f_y, f_x, f_z

    def get_delay_t(self, alpha):
        # alpha = self.alpha
        ndim = self.ndim
        d_alpha = [None, None]
        h = 1e-15

        def unit_vec(j): return np.array(
            [1 if i == j else 0 for i in range(ndim)])

        def alpha_t(t, y):
            return (alpha(t, y) - alpha(t - h, y))/h

        if self.n_state_delays == 1:
            def alpha_y(t, y):
                val = np.zeros(self.ndim, dtype=float)
                for j in range(ndim):
                    val[j] = (alpha(t, y) - alpha(t, y - h*unit_vec(j)))/h
                return np.atleast_1d(val)
        else:
            def alpha_y(t, y):
                delays = np.empty((self.n_state_delays, self.ndim), dtype=y.dtype)
                for i in range(self.n_state_delays):
                    val = np.zeros(self.ndim, dtype=float)
                    for j in range(ndim):
                        val[j] = (alpha(t, y)[i] -
                                  alpha(t, y - h*unit_vec(j))[i])/h
                    delays[i] = val
                return np.squeeze(delays)

        return alpha_t, alpha_y


class Solution:
    def __init__(self, problem: Problem):
        self.problem = problem
        self.t = [problem.t_span[0]]
        self.y = [np.atleast_1d(problem.phi(problem.t_span[0]))]
        self.etas = [problem.phi]
        self.etas_t = [problem.d_phi]
        self.etas_tt = [problem.d_phi_t]
        self.discs = [problem.t_span[0]]
        self.eta_calls = 0
        self.eta_t_calls = 0
        self.t_next = None
        # eta = self.build_eta()
        # self.eta = eta

    @property
    def eta(self):
        def eval(t):
            self.eta_calls += 1
            t = np.atleast_1d(t)  # accept scalar or array
            results = np.empty((len(t), self.problem.ndim), dtype=float)
            for i in range(len(t)):
                idx = bisect_left(self.t, t[i])
                if t[i] <= self.t[-1]:
                    results[i] = self.etas[idx](t[i])
                else:
                    raise ValueError(
                        f"eta isn't defined in {t[i]}, only on {self.t[0], self.t[-1]}")
            return np.squeeze(results)
        return eval

    @property
    def eta_t(self):
        def eval(t):
            self.eta_calls += 1
            t = np.atleast_1d(t)  # accept scalar or array
            results = np.empty((len(t), self.problem.ndim), dtype=float)
            for i in range(len(t)):
                idx = bisect_left(self.t, t[i])
                if t[i] <= self.t[-1]:
                    results[i] = self.etas_t[idx](t[i])
                else:
                    raise ValueError(
                        f"eta isn't defined in {t[i]}, only on {self.t[0], self.t[-1]}")
            return np.squeeze(results)
        return eval

    @property
    def eta_tt(self):
        def eval(t):
            self.eta_calls += 1
            t = np.atleast_1d(t)  # accept scalar or array
            results = np.empty((len(t), self.problem.ndim), dtype=float)
            for i in range(len(t)):
                idx = bisect_left(self.t, t[i])
                if t[i] <= self.t[-1]:
                    results[i] = self.etas_tt[idx](t[i])
                else:
                    raise ValueError(
                        f"eta isn't defined in {t[i]}, only on {self.t[0], self.t[-1]}")
            return np.squeeze(results)
        return eval

    def update(self, onestep):
        success, step = onestep
        if step.disc != False:
            self.discs.append(step.disc)

        if success:  # Step accepted
            # if (self.t[-1] + step.h != step.t[1]):
                # print('sum', self.t[-1] + step.h, 't1', step.t[1])
            self.t.append(step.t[0] + step.h)
            self.y.append(step.y[1])
            self.etas.append(step.new_eta[1])
            self.etas_t.append(step.new_eta_t[1])
            self.etas_tt.append(step.new_eta_tt)
            # h = step.h_next  # Use adjusted stepsize from rejection
            return None

        else:
            raise ValueError("Failed")
            return "Failed"


def solve_dde(f, alpha, phi, t_span, method = 'RK45', neutral=False, beta=None, d_f=None, d_alpha=None, d_phi=None):
    problem = Problem(f, alpha, phi, t_span, d_f = d_f, d_alpha = d_alpha,
                      d_phi = d_phi, beta = beta, neutral=neutral)
    solution = Solution(problem)
    params = CRKParameters()
    t, tf = problem.t_span

    h = (params.TOL ** (1 / 4)) * 0.1  # Initial stepsize
    print("-" * 80)
    print("Initial h:", h)
    print("-" * 80)

    # first_step = RK4HHL(problem, solution, h, neutral)
    first_step = RK4HHL(problem, solution, h, neutral)
    status = solution.update(first_step.first_step_CRK())
    if status != None:
        raise ValueError(status)
    h = first_step.h_next
    t = solution.t[-1]

    times = []
    calls = 0
    while t < tf:
        h = min(h, tf - t)
        # onestep = RK4HHL(problem, solution, h, neutral)
        onestep = RK4HHL(problem, solution, h, neutral)
        status = solution.update(onestep.one_step_CRK())
        calls += onestep.number_of_calls
        if status != None:
            raise ValueError(status)
        h = onestep.h_next
        t = solution.t[-1]

    return solution
