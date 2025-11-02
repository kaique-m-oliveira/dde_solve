import numbers
import random
import time
from bisect import bisect_left
from copy import deepcopy
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize, minimize_scalar, root


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
    else:
        f0 = f(t0, y0, eta(alpha0), eta(beta0))

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
    else:

        beta1 = beta(t0 + h0, y1)
        if np.all(beta1 <= t0):
            eta_beta1 = eta(beta1)
        else:
            eta_beta1 = y0 + ((beta1 - t0)/h0)*(y1 - y0)

        f1 = f(t0 + h0, y1, eta_alpha1, eta_beta1)

    d2 = norm((f1 - f0) / scale) / h0

    if d1 <= 1e-15 and d2 <= 1e-15:
        h1 = max(1e-6, h0 * 1e-3)
    else:
        h1 = (0.01 / max(d1, d2)) ** (1 / (order + 1))

    return min(100 * h0, h1, max_step)



def bisection_method(f, a, b, TOL):
    # print(f'old a = {a}, b  = {b}, tol = {TOL}')
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

    # input(f'new a = {a}, b  = {b}, tol = {TOL}')
    return [a, b]


@dataclass
class CRKParameters:
    theta1: float = 1 / 3
    # TOL: float = 1e-7
    rho: float = 0.9
    omega_min: float = 0.5  # was 0.5
    omega_max: float = 1.2  # between 1.5 and 5


def vectorize_func(func):
    def wrapper(*args, **kwargs):
        # return np.array(func(*args, **kwargs))
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

class Counting:
    steps = 0
    fails = 0
    fnc_calls = 0



class RungeKutta:
    def __init__(self, problem, solution, h, neutral=False, Atol=1e-8, Rtol=1e-8):

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
        self.h_next = None
        # Setting up self.t[1] = tn and self.y[1] = yn is a little hack to make self.investigate_branches()
        # work when there is for a breaking disc. is on the initial value t_span[0]
        self.t = [tn, tn]
        self.y = [yn, yn]
        self.n = problem.ndim
        self.y_tilde = None
        self.K = np.zeros((total_stages, self.n), dtype=float)
        self.eta = solution.eta
        self.eta_t = solution.eta_t
        self.new_eta = [None, None]
        self.new_eta_t = [None, None]
        self.disc_local_error = None
        self.uni_local_error = None
        self.params = CRKParameters()
        self.overlap = False
        self.test = False
        self.disc = None  # either False or a pair (disc_old, disc_new)
        self.ndim = problem.ndim
        self.ndelays = problem.n_state_delays
        self.fails = 0
        self.stages_calculated = 0
        self.store_times = []
        self.number_of_calls = 0
        self.neutral = neutral
        # self.Atol = np.full(self.y[0].shape, Atol)
        # self.Rtol = np.full(self.y[0].shape, Rtol)
        self.Atol = problem.Atol
        self.Rtol = problem.Rtol
        self.first_eta = True
        self.disc_position = False
        self.disc_beta_positions = False
        self.disc_interval = None
        self.breaking_step = False
        self.disc_flag = False

    @property
    def eeta(self):
        def eval(t):
            t = np.atleast_1d(t)  # accept scalar or array

            results = np.empty((len(t), self.problem.ndim), dtype=float)
            for i in range(len(t)):
                if t[i] <= self.t[0]:
                    results[i] = self.eta(t[i])
                else:
                    if self.new_eta[1] is not None:
                        results[i] = self.new_eta[1](t[i])
                    elif self.new_eta[0] is not None:
                        results[i] = self.new_eta[0](t[i])
                    elif not self.first_eta:
                        results[i] = self._hat_eta_0(t[i])
                    else:
                        results[i] = self.eta(t[i], ov=True)
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
                    elif not self.first_eta:
                        results[i] = self._hat_eta_0_t(t[i])
                    else:
                        results[i] = self.solution.eta_t(t[i], ov=True)
            return np.squeeze(results)
        return eval

    def is_there_disc(self):
        tn, h = self.t[0], self.h
        eta, alpha = self.solution.etas[-1], self.problem.alpha
        if self.neutral:
            beta = self.problem.beta
        discs = self.solution.discs

        if h <= 1e-12:
            return False

        def d_zeta(delay, t, disc):
            return delay(t, eta(t)) - disc  # np.full(self.ndelays, disc)

        for old_disc in discs:
            sign_change_alpha = d_zeta(
                alpha, tn, old_disc) * d_zeta(alpha, tn + h, old_disc) < 0
            if np.any(sign_change_alpha):
                self.disc_position = sign_change_alpha
                self.get_disc(alpha, old_disc)
                return True

            if self.neutral:
                sign_change_beta = d_zeta(
                    beta, tn, old_disc) * d_zeta(beta, tn + h, old_disc) < 0
                if np.any(sign_change_beta):
                    self.disc_position = sign_change_beta
                    self.get_disc(beta, old_disc)
                    return True

        return False

    def get_disc(self, delay, old_disc):
        indices = np.where(self.disc_position)[0].tolist()
        a, b = self.t[0], self.t[0] + self.h
        eta = self.solution.etas[-1]
        discs = []

        # discs almost never has more than one element
        for idx in indices[:]:
            def d_zeta_y1(t):
                self.h = t - self.t[0]
                self.one_step_RK4()
                y1 = self.y[1]
                return delay(t, y1)[idx] - old_disc

            def d_zeta(t):
                return delay(t, eta(t))[idx] - old_disc

            # We only need to check one disc
            self.disc_interval = bisection_method(
                d_zeta, a, b, TOL= np.min(self.Atol))
            self.old_disc = old_disc
            self.disc_delay_and_idx = (delay, idx)
            return

    def validade_disc(self):
        eta = self.new_eta[1]
        a, b = self.disc_interval
        old_disc = self.old_disc
        delay, idx = self.disc_delay_and_idx

        def d_zeta(t):
            return delay(t, eta(t))[idx] - old_disc

        tol = 0
        # input(f"is true disc at ({a}, {b + tol}) {d_zeta(a)*d_zeta(b + tol) < 0}")
        if d_zeta(a)*d_zeta(b + tol) < 0:
            return True
        else:
            return False

    def one_step_RK4(self, eta_ov=None, eta_t_ov=None):
        tn, h, yn = self.t[0], self.h, self.y[0]
        eta = self.eta
        f, alpha = self.problem.f, self.problem.alpha
        n_stages = self.n_stages["discrete_method"]
        c = self.c[:n_stages]
        A = self.A[:n_stages, :n_stages]
        if self.neutral:
            self.K[0] = f(tn, yn, eta(alpha(tn, yn)),
                          self.eta_t(self.problem.beta(tn, yn)))
            Counting.fnc_calls += 1
        else:
            self.K[0] = f(tn, yn, eta(alpha(tn, yn)))
            Counting.fnc_calls += 1
        self.stages_calculated = 1

        for i in range(1, n_stages):
            ti = tn + c[i] * h
            yi = yn + h * (A[i][0:i] @ self.K[0: i])

            alpha_i = alpha(ti, yi)

            if np.all(alpha_i <= np.full(self.ndelays, tn)):
                Y_tilde = eta(alpha_i)

            elif eta_ov is not None:
                Y_tilde = eta_ov(alpha_i)

            else:  # this would be the overlapping case
                self.overlap = True
                success = self.fixed_point()
                if not success:
                    return False
                break

            if not self.neutral:
                self.K[i] = f(ti, yi, Y_tilde)
                Counting.fnc_calls += 1
                self.stages_calculated = i + 1

            else:
                beta_i = self.problem.beta(ti, yi)

                if np.all(beta_i <= np.full(self.ndelays, tn)):
                    Z_tilde = self.solution.eta_t(beta_i)

                elif eta_t_ov is not None:
                    Z_tilde = eta_ov(beta_i)

                else:  # this would be the overlapping case
                    self.overlap = True
                    success = self.fixed_point()
                    if not success:
                        return False
                    break

                self.K[i] = f(ti, yi, Y_tilde, Z_tilde)
                Counting.fnc_calls += 1
                self.stages_calculated = i + 1

        self.y[1] = yn + h * (self.b @ self.K[0:n_stages])
        self.stages_calculated = n_stages

        if np.isnan(self.y[1]).any():
            print('K', self.K)
            print('shape', self.y[1].shape)
            input(f'y1 {self.y[1]}')

        return True

    def fixed_point(self):
        K = self.K[0:self.n_stages["discrete_method"]]
        self.one_step_RK4(eta_ov=self.eeta, eta_t_ov=self.eeta_t)
        self.first_eta = False
        max_iter = 10
        for i in range(max_iter):
            self.one_step_RK4(eta_ov=self.eeta, eta_t_ov=self.eeta_t)
            # if np.linalg.norm(K - self.K[0:self.n_stages["discrete_method"]]) <= 1e-7:
            #     return True
            sc = self.Atol + np.abs(self.y[0])*self.Rtol
            errs = np.linalg.norm((K - self.K[0:self.n_stages["discrete_method"]])/sc, axis = 1/np.sqrt(self.ndim))
            if np.max(errs) <= 1:
                return True
            K = self.K[0:self.n_stages["discrete_method"]]
        return False

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
                    Counting.fnc_calls += 1
                else:
                    self.K[i] = f(ti, yi, Y_tilde)
                    Counting.fnc_calls += 1
            self.stages_calculated = self.n_stages["continuous_err_est_method"]

    def _eta_0(self, theta):
        tn, h, yn = self.t[0], self.h, self.y[0]
        theta = (theta - tn) / h

        pol_order = self.D_err.shape[1]
        theta = theta ** np.arange(pol_order)
        K = self.K[0:self.n_stages["continuous_err_est_method"]]
        bs = (self.D_err @ theta).squeeze()
        eta0 = yn + h * bs @ K

        return eta0

    def _eta_0_t(self, theta):
        tn, h, yn = self.t[0], self.h, self.y[0]
        theta = (theta - tn) / h
        # pol_order = self.D_err.shape[1]
        # theta = np.array([n*theta**(n-1) for n in range(pol_order)])

        pol_order = self.D_err.shape[1]
        n = np.arange(pol_order)
        theta = np.where(n == 0, 0.0, n * theta ** (n - 1))
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
        tn, h, yn = self.t[0], self.h, self.y[0]
        theta = (theta - tn) / h
        # pol_order = self.D_ovl.shape[1]
        # theta = np.array([n*theta**(n-1) for n in range(pol_order)])

        pol_order = self.D_ovl.shape[1]
        n = np.arange(pol_order)
        theta = np.where(n == 0, 0.0, n * theta ** (n - 1))
        K = self.K[0:self.n_stages["continuous_ovl_method"]]
        bs = (self.D_ovl @ theta).squeeze()
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
                    Counting.fnc_calls += 1
                else:
                    self.K[i] = f(ti, yi, Y_tilde)
                    Counting.fnc_calls += 1
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
        n = np.arange(pol_order)
        theta = np.where(n == 0, 0.0, n * theta ** (n - 1))
        # theta = np.array([n*theta**(n-1) for n in range(pol_order)])
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
                    Counting.fnc_calls += 1
                else:
                    self.K[i] = f(ti, yi, Y_tilde)
                    Counting.fnc_calls += 1

            self.stages_calculated = self.n_stages["discrete_err_est_method"]

        K = self.K[0:self.n_stages["discrete_err_est_method"]]
        self.y_tilde = self.y[0] + self.h * (self.b_err @ K)

    def discrete_disc_satistied(self):
        sc = self.Atol + np.abs(self.y[0])*self.Rtol

        self.disc_local_error = (
            np.linalg.norm(
                (self.y_tilde - self.y[1])/sc)/np.sqrt(self.ndim)
        )  # eq 7.3.4

        if self.disc_local_error <= 1:
            return True
        else:
            return False


    # def look_for_max(self):
    #     # Define the function in terms of theta
    #     def diff_norm(theta):
    #         t_eval = self.t[0] + theta * self.h  # tn + θh
    #         return np.linalg.norm(self.new_eta[0](t_eval) - self.new_eta[1](t_eval))
    #     
    #     # Find maximum over theta in [0,1]
    #     result = minimize_scalar(lambda theta: -diff_norm(theta), 
    #                            bounds=(0, 1), options={"xatol": 1e-12, "maxiter": 500},
    #                            method='bounded')
    #     
    #     theta_max = result.x
    #     max_value = diff_norm(theta_max)
    #     t_max = self.t[0] + theta_max * self.h
    #     input(f'Maximum at θ = {theta_max:.4f}, t = {t_max:.4f}, value = {max_value:.6f}')

    def look_for_max(self, n_refinements=3):
        def diff_norm(theta):
            t_eval = self.t[0] + theta * self.h
            return np.linalg.norm(self.new_eta[0](t_eval) - self.new_eta[1](t_eval))
        
        # Iterative refinement
        search_range = (0, 1)
        
        for i in range(n_refinements):
            theta_samples = np.linspace(search_range[0], search_range[1], 100)
            norms = [diff_norm(theta) for theta in theta_samples]
            best_idx = np.argmax(norms)
            best_theta = theta_samples[best_idx]
            
            # Refine search range around current best
            if i < n_refinements - 1:  # Don't refine on last iteration
                span = (search_range[1] - search_range[0]) / 10
                search_range = (max(0, best_theta - span), min(1, best_theta + span))
        
        # Final local optimization
        result = minimize_scalar(lambda theta: -diff_norm(theta),
                               bounds=(search_range[0], search_range[1]),
                               method='bounded',
                               options={'xatol': 1e-12})
        
        final_theta = result.x
        final_max = diff_norm(final_theta)
        
        # input(f"Final maximum: θ = {final_theta:.10f}, value = {final_max:.10f}")



    def uniform_disc_satistied(self):

        tn, h = self.t[0], self.h
        val1 = self.new_eta[0](tn + h/2)
        val2 = self.new_eta[1](tn + h/2)
        sc = self.Atol + np.abs(self.y[0])*self.Rtol

        # self.look_for_max()

        self.uni_local_error = (
            np.linalg.norm((val1 - val2)/sc)/np.sqrt(self.ndim)
        )  # eq 7.3.4

        if self.uni_local_error <= 1:
            return True
        else:
            return False

    def try_step_CRK(self):
        print('______________________________________________________________')
        print('t = ', [self.t[0], self.t[0] + self.h], 'h = ', self.h)
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
        self.error_est_method()

        discrete_disc_satisfied = self.discrete_disc_satistied()

        uniform_disc_satistied = self.uniform_disc_satistied()

        facmax = self.params.omega_max
        facmin = self.params.omega_min
        fac = self.params.rho
        err1 = self.disc_local_error if self.disc_local_error >= 1e-15 else 1e-15
        err2 = self.uni_local_error if self.uni_local_error >= 1e-15 else 1e-15
        pp = min(self.order["discrete_method"],
                 self.order["discrete_err_est_method"])
        qq = min(self.order["continuous_method"],
                 self.order["continuous_err_est_method"])

        self.t[1] = self.t[0] + self.h
        self.h_next = self.h * \
            min(facmax, max(facmin, fac*min((1/err1) **
                (1/pp + 1), (1/err2)**(1/qq + 1))))
        # print('disc err = ', self.disc_local_error, 'uni err', self.uni_local_error)

        if not discrete_disc_satisfied or not uniform_disc_satistied:
            self.h = self.h_next
            print(f'not sucessfull disc satisfied: {
                  discrete_disc_satisfied} uni satisf: {uniform_disc_satistied}')
            Counting.fails += 1
            Counting.steps += 1
            return False

        err3 = np.linalg.norm(self.y[1] - self.y_tilde)/self.h

        Counting.steps += 1
        return True

    def investigate_disc(self):
        true_disc = self.validade_disc()

        if not true_disc:
            self.disc = None
            return

        self.disc = self.disc_interval[1]
        self.h = self.disc - self.t[0]

        # WARN: this is only concerning state dependent for now
        if self.solution.breaking_discs:
            self.get_possible_branches()

    def get_possible_branches(self):
        """
        This function checks for candidates for possible branches, 
        it only needs to check for breaking discontinuities
        """


        a, b = self.disc_interval
        eta, alpha = self.new_eta[1], self.problem.alpha
        self.alpha_discs = np.full(self.problem.n_state_delays, None)
        if self.neutral:
            beta = self.problem.beta
            self.beta_discs = np.full(self.problem.n_state_delays, None)
        discs = self.solution.discs

        def d_zeta(delay, t, disc):
            return delay(t, eta(t)) - disc  # np.full(self.ndelays, disc)

        for disc in self.solution.breaking_discs:
            sign_change_alpha = d_zeta(
                alpha, a, disc) * d_zeta(alpha, b, disc) < 0
            if np.any(sign_change_alpha):
                indices = np.where(sign_change_alpha)[0].tolist()
                self.alpha_discs[indices] = disc
                self.breaking_step = True

            if self.neutral:
                sign_change_beta = d_zeta(
                    beta, a, disc) * d_zeta(beta, b, disc) < 0

                if np.any(sign_change_beta):
                    indices = np.where(sign_change_beta)[0].tolist()
                    self.beta_discs[indices] = disc
                    self.alpha_discs[indices] = disc
                    self.breaking_step = True

    def investigate_branches(self):
        if self.disc is None:
            return None
        print('===================== INVESTIGATE ===========================')
        print(f't = {self.t[0] + self.h}')
        disc = self.disc
        f = self.problem.f
        alpha = self.problem.alpha
        alpha_discs = self.alpha_discs
        old_disc = np.array([x if x is not None else 0 for x in alpha_discs])
        eta = self.solution.eta
        eps = np.finfo(float).eps**(1/3)

        alpha_limits = (self.alpha_discs != None) + 0
        idx = np.where(alpha_limits)[0]
        N = len(idx)

        continuation = []
        limit_directions = []
        for mask in range(1 << N):      # loop over 0..2^k-1
            limit_direction = alpha_limits.copy()
            for j in range(N):
                if (mask >> j) & 1:     # check j-th bit
                    limit_direction[idx[j]] = -1

            t1 = self.t[1]
            y1 = self.y[1]

            # FIX: gotta fix this thing man
            # if self.problem.n_state_delays == 1:
            #     alpha1 = alpha(t1, y1)
            # else:
            #     alpha1 = alpha(t1, y1).squeeze()

            alpha1 = alpha(t1, y1)
            alpha1 = [alpha1[i] if alpha_discs[i] is None else alpha_discs[i]
                      for i in range(len(alpha1))]


            if not self.neutral:
                print('limit_dir', limit_direction)
                limit_directions.append(limit_direction)
                y_lim = y1 + eps * \
                    f(t1, y1, eta(alpha1, limit_direction=limit_direction))

                print('t1', t1)
                print('y1', y1)
                print('eta(alpha)', eta(alpha1, limit_direction=limit_direction))
                print('y_lim', y_lim)
                print('t1 + eps', t1+eps)
                print('alpha', -1*limit_direction *(alpha(t1 + eps, y_lim) - old_disc) )
                continued = -1*limit_direction * \
                    (alpha(t1 + eps, y_lim) - old_disc) < 0
                mask = np.array(alpha_limits.astype(bool))

                continued = continued[mask]
                continuation.append(continued)
                print('___________________________________________________')

            else:
                eta_t = self.solution.eta_t
                print('limit_dir', limit_direction)
                limit_directions.append(limit_direction)
                y_lim = y1 + eps * \
                    f(t1, y1, eta(alpha1, limit_direction=limit_direction), eta_t(alpha1 + limit_direction*10**-16, limit_direction=limit_direction) )

                continued = -1*limit_direction * \
                    (alpha(t1 + eps, y_lim) - old_disc) < 0
                mask = np.array(alpha_limits.astype(bool))
                print('y_lim', y_lim)

                continued = continued[mask]
                continuation.append(continued)
                print('___________________________________________________')

        print('all discs', self.solution.discs)
        print('all breaking discs', self.solution.breaking_discs)
        print('counting steps', Counting.steps)
        print('counting fails', Counting.fails)
        print('continuation', continuation)
        input(f'ever here? t = {t1}')
        if not np.any(np.all(continuation, axis=1)):
            return "terminated"

        possible_branches = np.all(continuation, axis=1)
        print('possible_branches', possible_branches)
        if sum(possible_branches) == 1:
            pos = np.where(possible_branches)[0][0]
            self.limit_direction = limit_directions[pos]
            print('lit ditection', self.limit_direction)
            return "one branch"
        else:
            pos = np.where(possible_branches)[0].tolist()
            self.limit_directions = np.array(limit_directions)[pos]
            return "branches"

    def one_step_CRK(self, max_iter=10):
        iter = 0
        while self.h >= 10**-12 and iter <= max_iter:
            success = self.try_step_CRK()
            if success:
                return True, self

            disc_found = self.is_there_disc()
            if disc_found:
                h = self.disc_interval[0] - self.t[0]

                if h >= 1e-14:
                    self.h = h
                    success = self.try_step_CRK()
                    if success:
                        self.investigate_disc()
                        return True, self
            iter += 1

        return False, 0

    def first_step_investigate_branch(self):
        if self.solution.breaking_discs:
            self.alpha_discs = np.full(self.problem.n_state_delays, None)
            alpha0 = self.problem.alpha(self.t[0], self.y[0])
            print('alpha0', alpha0, 'shape', alpha0.shape)
            print('breaking discs', self.solution.breaking_discs)

            for i in range(self.problem.n_state_delays):
                if self.ndim == 1:
                    if float(alpha0[i]) in self.solution.breaking_discs:
                        self.alpha_discs[i] = float(alpha0[i])
                # WARN:  NO CLUE IF THIS WORKS
                else:
                    for dim in range(self.ndim):
                        if float(alpha0[i][dim]) in self.solution.breaking_discs:
                            self.alpha_discs[i] = float(alpha0[i][dim])

            if np.any(self.alpha_discs):
                self.disc = self.t[0]
                state = self.investigate_branches()
                return state

        return None




class RK3C(RungeKutta):
    A = np.array([
        [0, 0, 0, 0],
        [1/2, 0, 0, 0],
        [0, 3/4, 0, 0],
        [2/9, 1/3, 4/9, 0]
    ], dtype=np.float64)

    b = np.array([2/9, 1/3, 4/9], dtype=np.float64)
    b_err = np.array([7/24, 1/4, 1/3, 1/8], dtype=np.float64)
    c = np.array([0, 1/2, 3/4, 1], dtype=np.float64)

    D = np.array([
        [0.0, 1.0, -4/3,  5/9],     # d1(theta) = theta - 4/3 theta^2 + 5/9 theta^3
        [0.0, 0.0,  1.0, -2/3],     # d2(theta) = theta^2 - 2/3 theta^3
        [0.0, 0.0,  4/3, -8/9],     # d3(theta) = 4/3 theta^2 - 8/9 theta^3
        [0.0, 0.0, -1.0,  1.0]      # d4(theta) = -theta^2 + theta^3
        ], dtype=np.float64)

    D_err = D
    D_ovl = D

    order = {
        "discrete_method": 3,
        "discrete_err_est_method": 2,
        "continuous_method": 3,
        "continuous_err_est_method": 3,
        "continuous_ovl_method": 3
    }

    n_stages = {
        "discrete_method": 3,
        "discrete_err_est_method": 4,
        "continuous_method": 4,
        "continuous_err_est_method": 4,
        "continuous_ovl_method": 4
    }


class DormandPrince54(RungeKutta):
    A = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1/5, 0, 0, 0, 0, 0, 0, 0, 0],
        [3/40, 9/40, 0, 0, 0, 0, 0, 0, 0],
        [44/45, -56/15, 32/9, 0, 0, 0, 0, 0, 0],
        [19372/6561, -25360/2187, 64448/6561, -212/729, 0, 0, 0, 0, 0],
        [9017/3168, -355/33, 46732/5247, 49/176, -5103/18656, 0, 0, 0, 0],
        [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0, 0, 0],
        [-33728713/104693760, 2, -30167461/21674880, 7739027/17448960, -19162737/123305984, 0, -26949/363520, 0, 0],
        [7157/75776, 0, 70925/164724, 10825/113664, -220887/4016128, 80069/3530688, -107/5254, -5/74, 0]
        ], dtype=np.float64)

    b = np.array([35/384, 0, 500/1113, 125/192, -2187/6784, 11/84], dtype=np.float64)

    b_err = np.array([5179/57600, 0, 7571/16695, 393/640, -92097/339200, 187/2100, 1/40], dtype=np.float64)

    c = np.array([0, 1/5, 3/10, 4/5, 8/9, 1, 1, 1/2, 1/2], dtype=np.float64)

    D = np.array([
        [0, 1, -6839/1776, 24433/3552, -81685/14208, 29/16],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 413200/41181, -398800/13727, 1245700/41181, -4000/371],
        [0, 0, 225/37, -44725/1776, 83775/2368, -125/8],
        [0, 0, -98415/31376, 798255/62752, -4428675/251008, 6561/848],
        [0, 0, 23529/18389, -285659/55167, 527571/73556, -22/7],
        [0, 0, -3483/2627, 14847/2627, -21872/2627, 4],
        [0, 0, -40/37, 80/37, -40/37, 0],
        [0, 0, -8, 32, -40, 16]
        ], dtype=np.float64)

    D_err = np.array([
        [0, 1, -4034104133/1410260304, 105330401/33982176, -13107642775/11282082432, 6542295/470086768],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 132343189600/32700410799, -833316000/131326951, 91412856700/32700410799, -523383600/10900136933],
        [0, 0, -115792950/29380423, 185270875/16991088, -12653452475/1880347072, 98134425/235043384],
        [0, 0, 70805911779/24914598704, -4531260609/600351776, 988140236175/199316789632, -14307999165/24914598704],
        [0, 0, -331320693/205662961, 31361737/7433601, -2426908385/822651844, 97305120/205662961],
        [0, 0, 44764047/29380423, -1532549/353981, 90730570/29380423, -8293050/29380423]
    ], dtype=np.float64)

    D_ovl = D_err

    order = {
        "discrete_method": 5,
        "discrete_err_est_method": 4,
        "continuous_method": 4,
        "continuous_err_est_method": 4,
        "continuous_ovl_method": 4
    }

    n_stages = {
        "discrete_method": 6,
        "discrete_err_est_method": 7,   
        "continuous_method": 9,   
        "continuous_err_est_method": 7,
        "continuous_ovl_method": 7
    }




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


class Problem:
    def __init__(self, f, alpha, phi, t_span, Atol, Rtol, beta=False, phi_t=False, neutral=False):
        ndim, n_state_delays, n_neutral_delays, f, alpha, phi, t_span, beta, phi_t = validade_arguments(
            f, alpha, phi, t_span,  beta=beta, phi_t=phi_t)
        self.t_span = np.array(t_span)
        self.ndim, self.n_state_delays, self.n_neutral_delays = ndim, n_state_delays, n_neutral_delays
        self.f, self.alpha, self.phi, self.t_span = f, alpha, phi, t_span
        self.beta, self.phi_t = beta, phi_t
        self.y_type = np.zeros(self.ndim, dtype=float).dtype
        self.neutral = neutral
        self.Atol = np.full(ndim, Atol)
        self.Rtol = np.full(ndim, Rtol)


class Solution:
    def __init__(self, problem: Problem, discs=[], neutral=None):
        self.problem = problem
        self.t = [problem.t_span[0]]
        self.y = [np.atleast_1d(problem.phi(problem.t_span[0]))]
        self.etas = [problem.phi]
        self.etas_t = [problem.phi_t]

        self.breaking_discs = {}
        self.phi_t_breaks = {}
        if discs:
            self.validade_discs(discs)
        else:
            self.discs = [problem.t_span[0]]

        self.eta_calls = 0
        self.eta_t_calls = 0
        self.t_next = None
        self.neutral = neutral

    @property
    def eta(self, ov=False, limit_direction=None):
        def eval(t, ov=ov, limit_direction=limit_direction):
            self.eta_calls += 1
            t = np.atleast_1d(t)  # accept scalar or array
            results = np.empty((len(t), self.problem.ndim), dtype=float)
            for i in range(len(t)):
                idx = bisect_left(self.t, t[i])
                if t[i] <= self.t[0]:
                    if limit_direction is not None:
                        if limit_direction[i] != 0:
                            if t[i] in self.breaking_discs:
                                disc = self.breaking_discs[t[i]]
                                results[i] = disc[limit_direction[i]]
                                continue
                    results[i] = self.etas[0](t[i])
                elif t[i] <= self.t[-1]:
                    results[i] = self.etas[idx](t[i])
                else:
                    if ov:
                        results[i] = self.etas[-1](t[i])
                    else:
                        raise ValueError(
                            f"eta isn't defined in {t[i]}, only on {self.t[0], self.t[-1]}")
            return np.squeeze(results)
        return eval

    @property
    def eta_t(self, ov=False, limit_direction=None):
        def eval(t, ov=ov, limit_direction=limit_direction):
            self.eta_calls += 1
            t = np.atleast_1d(t)  # accept scalar or array
            results = np.empty((len(t), self.problem.ndim), dtype=float)
            for i in range(len(t)):
                idx = bisect_left(self.t, t[i])
                if t[i] <= self.t[0]:
                    if limit_direction is not None:
                        if limit_direction[i] != 0:
                            if t[i] in self.phi_t_breaks:
                                disc = self.phi_t_breaks[t[i]]
                                results[i] = disc[limit_direction[i]]
                                continue
                            # WARN: Special consideration for the first step
                            elif t[i] == self.t[0]:
                                if limit_direction[i] == -1:
                                    results[i] = self.etas_t[0](self.t[0])
                                    continue
                                if limit_direction[i] == 1:
                                    results[i] = self.etas_t[1](self.t[0])
                                    continue
                    results[i] = self.etas_t[0](t[i])
                elif t[i] <= self.t[-1]:
                    results[i] = self.etas_t[idx](t[i])
                else:
                    if ov:
                        results[i] = self.etas_t[-1](t[i])
                    else:
                        print('Counting.steps', Counting.steps)
                        print('Counting.fails', Counting.fails)
                        raise ValueError(
                            f"eta isn't defined in {t[i]}, only on {self.t[0], self.t[-1]}")
            return np.squeeze(results)
        return eval

    def validade_discs(self, discs):
        if not isinstance(discs, (list, tuple, np.ndarray)):
            raise TypeError("discs should be a list of tuples")

        for disc in discs:
            if not isinstance(disc, (list, tuple, np.ndarray)):
                raise TypeError("discs should be a list of tuples")

            if len(disc) != 3:
                raise TypeError(
                    f" Problem with one of the discontinuities, lenght of {disc} is not 3")

            # Validating discontinuity
            if disc[0] > self.problem.t_span[0]:
                raise ValueError(
                    f"Discontinuities beyond t_span[0] are not allowed")

            # Validading left limit
            if isinstance(disc[1], numbers.Real) or np.isscalar(disc[1]):
                if 1 != self.problem.ndim:
                    raise TypeError(
                        f"Dimension of one the left limits doesn't math dimension of phi")
            elif isinstance(disc[1], (list, np.ndarray)):
                if len(disc[1]) != self.problem.ndim:
                    raise TypeError(
                        f"Dimension of one the left limits doesn't math dimension of phi")
            else:
                raise TypeError(
                    f"Unsupported type left limit:{type(disc[1])}")

            # Validading right limit
            if isinstance(disc[2], numbers.Real) or np.isscalar(disc[1]):
                if 1 != self.problem.ndim:
                    raise TypeError(
                        f"Dimension of one the right limits doesn't math dimension of phi")
            elif isinstance(disc[2], (list, np.ndarray)):
                if len(disc[2]) != self.problem.ndim:
                    raise TypeError(
                        f"Dimension of one the right limits doesn't math dimension of phi")
            else:
                raise TypeError(
                    f"Unsupported type right limit:{type(disc[1])}")

        discs.sort(key=lambda x: x[0])

        for disc in discs:
            self.breaking_discs[disc[0]] = {-1: disc[1], 1: disc[2]}

        self.discs = [x[0] for x in discs]
        if self.problem.t_span[0] not in self.discs:
            self.discs.append(self.problem.t_span[0])

    def update(self, onestep):
        success, step = onestep

        if success:
            self.t.append(step.t[0] + step.h)
            self.y.append(step.y[1])
            self.etas.append(step.new_eta[1])
            self.etas_t.append(step.new_eta_t[1])

            if step.disc:
                self.discs.append(step.disc)
                # input(f'discs {self.discs}, breaking discs = {self.breaking_discs}')
                if step.breaking_step:
                    self.breaking_discs[step.disc] = {-1 : -1, 1 : 1}
                    progress = step.investigate_branches()
                    if progress == "terminated":
                        return "terminated"
                    elif progress == "one branch":
                        return "one branch"
                    elif progress == "branches":
                        self.limit_directions = step.limit_directions
                        return "branches"

        else:
            return "Failed"

        return "Success"


class SolutionList:
    def __init__(self):
        self.solutions = []

    def append_solution(self, sol):
        self.solutions.append(sol)


def recursive_integration(solution, solutionList):
    for limit_direction in solution.limit_directions:
        solution_copy = deepcopy(solution)
        status, solution_copy = integrate_branch(
            solution_copy, limit_direction)
        if status == "branches":
            recursive_integration(solution_copy, solutionList)
        else:
            solutionList.append_solution(solution_copy)
            print('solutionList', solutionList.solutions)


def integrate_branch(solution, limit_direction):
    t, tf = solution.t[-1], solution.problem.t_span[-1]
    problem = solution.problem
    neutral = solution.neutral
    h = (1e-7 ** (1 / 4)) * 0.1  # Initial stepsize

    # WARN: first integration step will be a limit_direction integration
    onestep = RK4HHL(problem, solution, h, neutral)
    onestep.eta = lambda t: solution.eta(
        t, limit_direction=limit_direction)

    status = solution.update(onestep.one_step_CRK())

    calls = 0
    while t < tf:
        h = min(h, tf - t)
        if status == "Success":
            onestep = RK4HHL(problem, solution, h, neutral)

        elif status == "one branch":
            limit_direction = onestep.limit_direction
            onestep = RK4HHL(problem, solution, h, neutral)
            onestep.eta = lambda t: solution.eta(
                t, limit_direction=limit_direction)

        elif status == "branches":
            return "branches", solution

        elif status == "terminated" or status == "failed":
            raise ValueError(f"solution failed duo to {status} at t = {solution.t[-1]}")

        status = solution.update(onestep.one_step_CRK())
        calls += onestep.number_of_calls
        h = onestep.h_next
        t = solution.t[-1]
    return "Success", solution


def solve_dde(f, alpha, phi, t_span, method='RK45', Atol = 1e-7, Rtol = 1e-7, neutral=False, beta=None, d_phi=None, discs=[]):
    problem = Problem(f, alpha, phi, t_span, Atol, Rtol, beta=beta,
                      phi_t=d_phi, neutral=neutral)
    solution = Solution(problem, discs=discs, neutral=neutral)
    params = CRKParameters()
    t, tf = problem.t_span


    # h = (np.min(Atol)** (1 / 4)) * 0.1  # Initial stepsize
    order = 5
    h = get_initial_step(problem, solution, Atol, Rtol, order, neutral = neutral)
    # h = 1e-3
    print("-" * 80)
    print("Initial h:", h)
    print("-" * 80)

    # onestep = RK4HHL(problem, solution, h, neutral)
    onestep = DormandPrince54(problem, solution, h, neutral)
    # onestep = RK3C(problem, solution, h, neutral)
    branch_status = onestep.first_step_investigate_branch()

    if branch_status == "one branch":
        limit_direction = onestep.limit_direction
        onestep.eta = lambda t: solution.eta(
            t, limit_direction=limit_direction)
    elif branch_status == "branches":
        solutionList = SolutionList()
        solution.limit_directions = onestep.limit_directions
        recursive_integration(solution, solutionList)
        return solutionList
    elif branch_status == "terminated" or branch_status == "failed":
        raise ValueError(f"solution failed duo to {branch_status}")

    status = solution.update(onestep.one_step_CRK())

    h = onestep.h_next
    t = solution.t[-1]

    times = []
    calls = 0
    while t < tf:
        h = min(h, tf - t)
        if status == "Success":
            # onestep = RK4HHL(problem, solution, h, neutral)
            onestep = DormandPrince54(problem, solution, h, neutral)
            # onestep = RK3C(problem, solution, h, neutral)
        elif status == "one branch":
            limit_direction = onestep.limit_direction
            # onestep = RK4HHL(problem, solution, h, neutral)
            onestep = DormandPrince54(problem, solution, h, neutral)
            # onestep = RK3C(problem, solution, h, neutral)
            onestep.eta = lambda t: solution.eta(
                t, limit_direction=limit_direction)
        elif status == "branches":
            solutionList = SolutionList()
            input('here')
            recursive_integration(solution, solutionList)
            return solutionList
        elif status == "terminated" or status == "failed":
            print('Counting.steps', Counting.steps)
            print('Counting.fails', Counting.fails)
            print('Counting.fnc_calls', Counting.fnc_calls)
            raise ValueError(f"solution failed duo to {status} at t = {solution.t[-1]}")

        status = solution.update(onestep.one_step_CRK())
        calls += onestep.number_of_calls
        h = onestep.h_next
        t = solution.t[-1]

    return solution


# def solve_dde(f, alpha, phi, t_span, method='RK45', neutral=False, beta=None, d_phi=None, discs=[]):
def solve_ndde(t_span, f, alpha, beta, phi, phi_t, method='RK45', discs=[], Atol=1e-7, Rtol=1e-7):
    problem = Problem(f, alpha, phi, t_span, Atol = Atol, Rtol = Rtol, beta = beta, phi_t = phi_t, neutral=True)
    solution = Solution(problem, discs=discs, neutral=True)
    params = CRKParameters()
    t, tf = problem.t_span

    order = 5
    h = get_initial_step(problem, solution, Atol, Rtol, order, neutral = True)
    # h = (np.max(Atol) ** (1 / 4))  # Initial stepsize
    # input(f'Atol Rtol', problem.Atol, problem.Rtol)
    print("-" * 80)
    print("Initial h:", h)
    print("-" * 80)

    # onestep = RK4HHL(problem, solution, h, neutral=True)
    onestep = DormandPrince54(problem, solution, h, neutral=True)
    # onestep = RK3C(problem, solution, h, neutral=True)
    branch_status = onestep.first_step_investigate_branch()

    if branch_status == "one branch":
        limit_direction = onestep.limit_direction
        onestep.eta = lambda t: solution.eta(
            t, limit_direction=limit_direction)
    elif branch_status == "branches":
        solutionList = SolutionList()
        solution.limit_directions = onestep.limit_directions
        recursive_integration(solution, solutionList)
        return solutionList
    elif branch_status == "terminated" or branch_status == "failed":
        raise ValueError(f"solution failed duo to {branch_status}")

    status = solution.update(onestep.one_step_CRK())
    eta_t_left = solution.eta_t(t_span[0], limit_direction=[-1])
    eta_t_right = solution.eta_t(t_span[0], limit_direction=[1])
    if abs(eta_t_left - eta_t_right) >= 100*np.max(Atol):
        solution.breaking_discs[t_span[0]] = {-1: eta_t_left, 1: eta_t_right}
        print('solution.discs', solution.discs)
        print('eta_t_left', eta_t_left)
        print('eta_t_right', eta_t_right)

    # WARN: We need to verify now if the first step is a breaking step


    h = onestep.h_next
    t = solution.t[-1]

    times = []
    calls = 0
    while t < tf:
        h = min(h, tf - t)
        if status == "Success":
            # onestep = RK4HHL(problem, solution, h, neutral=True)
            onestep = DormandPrince54(problem, solution, h, neutral=True)
            # onestep = RK3C(problem, solution, h, neutral=True)
        elif status == "one branch":
            limit_direction = onestep.limit_direction
            # onestep = RK4HHL(problem, solution, h, neutral=True)
            onestep = DormandPrince54(problem, solution, h, neutral=True)
            # onestep = RK3C(problem, solution, h, neutral=True)
            onestep.eta = lambda t: solution.eta(
                t, limit_direction=limit_direction)
        elif status == "branches":
            solutionList = SolutionList()
            input('here')
            recursive_integration(solution, solutionList)
            return solutionList
        elif status == "terminated" or status == "failed":
            print('Counting.steps', Counting.steps)
            print('Counting.fails', Counting.fails)
            print('Counting.fnc_calls', Counting.fnc_calls)
            raise ValueError(f"solution failed duo to {status} at t = {solution.t[-1]}")



        status = solution.update(onestep.one_step_CRK())
        calls += onestep.number_of_calls
        h = onestep.h_next
        t = solution.t[-1]

        # eta_t_left = solution.eta_t(solution.t[-2] - 10**5* np.finfo(float).eps, limit_direction=[-1])
        # eta_t_right = solution.eta_t(solution.t[-2] + 10**5*np.finfo(float).eps , limit_direction=[1])
        # print('t = ', solution.t[-2])
        # print('eta_t_left', eta_t_left)
        # print('eta_t_right', eta_t_right)
        # # input('difs')
        # print('abs val', abs(eta_t_left - eta_t_right) >= 100*np.max(Atol))
        # if abs(eta_t_left - eta_t_right) >= 100*np.max(Atol):
        #     solution.breaking_discs[solution.t[-2]] = {-1: eta_t_left, 1: eta_t_right}
        #     print('solution.discs', solution.discs)
        #     print('eta_t_left', eta_t_left)
        #     print('eta_t_right', eta_t_right)
        #     print('solution.breaking_discs', solution.breaking_discs)
        #     # input('added')


    return solution
