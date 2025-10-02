import time
import numbers
from bisect import bisect_left
from dataclasses import dataclass, field
import random
import matplotlib.pyplot as plt
import numpy as np
# from scipy.optimize import root
from scipy.integrate import solve_ivp


def interval_bisection_step(f, a, b, TOL=1e-8, iter_max=100):
    a_new = a
    fa_new = f(a)
    b_new = b
    fb_new = f(b)
    iter = 0

    if fa_new * fb_new > 0:
        raise ValueError("this shouldn't be happening in the bisection step")

    while a_new == a or b_new == b:
        c = (a_new + b_new) / 2
        fc = f(c)

        if fc == 0:
            return [c - TOL/2, c + TOL/2]

        if fa_new*fc < 0:
            b_new = c
            fb_new = fc
        else:
            a_new = c
            fa_new = fc

    return [a_new, b_new]


@dataclass
class CRKParameters:
    theta1: float = 1 / 3
    TOL: float = 1e-7
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
    elif isinstance(alpha0, (list, np.ndarray)):
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
        self.disc_local_error = None
        self.uni_local_error = None
        self.params = CRKParameters()
        self.overlap = False
        self.test = False
        self.disc = False  # either False or a pair (disc_old, disc_new)
        self.ndim = problem.ndim
        self.ndelays = problem.n_state_delays
        self.old_disc = np.full(problem.n_state_delays, None)
        self.fails = 0
        self.stages_calculated = 0
        self.store_times = []
        self.number_of_calls = 0
        self.neutral = neutral
        self.Atol = np.full(self.y[0].shape, Atol)
        self.Rtol = np.full(self.y[0].shape, Rtol)
        self.first_eta = True
        self.disc_position = False
        self.disc_beta_positions = False
        self.disc_interval = None

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
                    elif not self.first_eta:
                        results[i] = self._hat_eta_0(t[i])
                    else:
                        results[i] = self.solution.eta(t[i], ov=True)
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
                # else:
                #     results[i] = self._hat_eta_0_t(t[i])
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
            new_disc = None
            if np.any(sign_change_alpha):
                self.disc_position = sign_change_alpha
                new_disc = self.step_with_disc(alpha, old_disc)
                if new_disc:
                    self.disc = self.t[0] + self.h
                return True

            if self.neutral:
                sign_change_beta = d_zeta(
                    beta, tn, old_disc) * d_zeta(beta, tn + h, old_disc) < 0
                if np.any(sign_change_beta):
                    d_beta = self.problem.d_beta
                    new_disc_beta = self.get_disc(
                        beta, d_beta, old_disc, sign_change_beta)
                    if new_disc is not None:
                        new_disc = min(new_disc, new_disc_beta)

            if new_disc is not None:
                self.disc = (old_disc, new_disc)
                return True
        return False

    def step_with_disc(self, delay, disc):
        eta = self.solution.etas[-1]
        print('pos', self.disc_position)
        indices = np.where(self.disc_position)[0].tolist()
        a, b = self.t[0], self.t[0] + self.h
        TOL = np.max(self.Atol)

        for idx in indices[:]:

            def d_zeta(t):
                return delay(t, eta(t))[idx] - disc

            if self.disc_interval is None:
                max_iter = int(np.ceil(np.log2(self.h/TOL))) + 3
                for _ in range(max_iter):
                    if (b - a)/2 < TOL:
                        self.h = b - self.t[0]
                        self.disc_interval = [a, b]
                        break

                    a_new, b = interval_bisection_step(
                        d_zeta, a, b, TOL=np.max(self.Atol), iter_max=100)
                    self.h = a_new - self.t[0]
                    success = self.try_step_CRK()
                    if success:
                        eta = self.new_eta[1]
                        a = a_new
                    else:
                        self.h = (a_new - a)/2 - self.t[0]
                        success = self.try_step_CRK()
                        if success:
                            eta = self.new_eta[1]
                            a = (a_new - a)/2
                        else:
                            # Failed to spot a disc, gotta remove it
                            indices.remove(idx)
                            break
            else:
                a, b = self.disc_interval
                # we now only need to check if the new idx fails
                if d_zeta(a)*d_zeta(b) >= 0:
                    indices.remove(idx)

        # if we got here, we found the disc at the indices
        self.old_disc[indices] = disc

        return True

    def one_step_RK4(self, eta_ov=None, eta_t_ov=None):
        tn, h, yn = self.t[0], self.h, self.y[0]
        eta = self.solution.eta
        f, alpha = self.problem.f, self.problem.alpha
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
                if self.neutral:
                    beta_i = self.problem.beta(ti, yi)
                    Z_tilde = self.eta_t(alpha_i)
                    self.K[i] = f(ti, yi, Y_tilde, Z_tilde)
                else:
                    self.K[i] = f(ti, yi, Y_tilde)
                self.stages_calculated = i + 1

            elif eta_ov is not None:
                alpha_i = alpha(ti, yi)
                Y_tilde = eta_ov(alpha_i)
                if self.neutral:
                    beta_i = self.problem.beta(ti, yi)
                    Z_tilde = eta_t_ov(alpha_i)
                    self.K[i] = f(ti, yi, Y_tilde, Z_tilde)
                else:
                    self.K[i] = f(ti, yi, Y_tilde)
                self.stages_calculated = i + 1

            else:  # this would be the overlapping case
                self.overlap = True
                success = self.fixed_point()
                if not success:
                    return False
                break

        self.y[1] = yn + h * (self.b @ self.K[0:n_stages])
        self.stages_calculated = n_stages
        # if self.t[0] >= 0.9983319415609049:
        if np.isnan(self.y[1]).any():
            print('K', self.K)
            print('shape', self.y[1].shape)
            input(f'y1 {self.y[1]}')

        return True

    def fixed_point(self):
        tn, h = self.t[0], self.h
        alpha = self.problem.alpha

        if self.neutral:
            beta = self.problem.beta

        K = self.K[0:self.n_stages["discrete_method"]]
        self.one_step_RK4(eta_ov=self.eeta, eta_t_ov=self.eeta_t)
        self.first_eta = False
        max_iter = 10
        for i in range(max_iter):
            self.one_step_RK4(eta_ov=self.eeta, eta_t_ov=self.eeta_t)
            if np.linalg.norm(K - self.K[0:self.n_stages["discrete_method"]]) <= 1e-7:
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
                else:
                    self.K[i] = f(ti, yi, Y_tilde)
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
        tn, h, yn = self.t[0], self.h, self.y[0]
        theta = (theta - tn) / h
        pol_order = self.D_ovl.shape[1]
        theta = np.array([n*theta**(n-1) for n in range(pol_order)])
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

    def discrete_disc_satistied(self):
        sc = self.Atol + \
            np.maximum(np.abs(self.y[1]), np.abs(self.y_tilde))*self.Rtol

        self.disc_local_error = (
            np.linalg.norm(
                (self.y_tilde - self.y[1])/sc)/np.sqrt(self.ndim)  # /self.h
        )  # eq 7.3.4

        if self.disc_local_error <= 1:
            return True
        else:
            return False

    def uniform_disc_satistied(self):
        # print('_______________________uni__________________________')

        tn, h = self.t[0], self.h
        val1 = self.new_eta[0](tn + h/2)
        val2 = self.new_eta[1](tn + h/2)
        sc = self.Atol + np.maximum(np.abs(val1), np.abs(val2))*self.Rtol

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
        pp = 4  # FIX: adicionar o agnóstico
        qq = 3
        self.h_next = self.h * \
            min(facmax, max(facmin, min(fac*(1/err1) **
                (1/pp + 1), fac*(1/err2)**(1/qq + 1))))

        # print('disc err = ', self.disc_local_error, 'uni err', self.uni_local_error)

        if not discrete_disc_satisfied or not uniform_disc_satistied:
            self.h = self.h_next
            print(f'not sucessfull disc satisfied: {
                  discrete_disc_satisfied} uni satisf: {uniform_disc_satistied}')
            return False

        err3 = np.linalg.norm(self.y[1] - self.y_tilde)/self.h

        return True

    def termination_test(self):
        epsilon = np.sqrt(np.finfo(float).eps)
        alpha = self.problem.alpha
        y1 = self.y[1]
        print('y1', y1)
        f = self.problem.f
        eta = self.eeta
        print('old_disc', self.old_disc)
        old_disc = self.old_disc[0]
        t = self.disc

        print('disc_position', self.disc_position)
        print('old_disc', old_disc)

        y_minus = y1 + epsilon*f(t, y1, eta(old_disc - epsilon))
        print('-----------------y_minus-----------------')
        print('f()', f(t, y1, eta(old_disc - epsilon)))
        print('eps*f()', epsilon * f(t, y1, eta(old_disc - epsilon)))
        print('y_minus', y_minus)

        y_plus = y1 + epsilon*f(t, y1, eta(old_disc + epsilon))
        print('-----------------y_plus-----------------')
        print('f()', f(t, y1, eta(old_disc + epsilon)))
        print('eps*f()', epsilon * f(t, y1, eta(old_disc + epsilon)))
        print('y_plus', y_plus)

        plus = alpha(t + epsilon, y_plus) > old_disc
        minus = alpha(t + epsilon, y_minus) < old_disc
        print('---------------------alpha----------------------')
        print('disc', self.t[0] + self.h)
        print(f'plus {plus}')
        print(f'minus {minus}')

        print('--------------------D stuff --------------------')
        D_plus = alpha(t + epsilon, y_plus) - alpha(t, y1)
        D_minus = alpha(t + epsilon, y_minus) - alpha(t, y1)
        print(f'D_plus {D_plus} is less then zero {D_plus < 0}')
        input(f'D_minus {D_minus} is more than zero {D_minus > 0}')

    def one_step_CRK(self, max_iter=15):
        success = self.try_step_CRK()

        if success:
            return True, self
        else:
            for i in range(max_iter - 1):
                if self.h <= 10**-12:
                    return False, 0

                disc_found = self.is_there_disc()

                if disc_found:
                    self.termination_test()
                    return True, self

                success = self.try_step_CRK()
                if success:
                    return True, self

            return False, 0

    def first_step_CRK(self, max_iter=5):
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
        [0.,  1., -2.86053867,  3.09957788, -1.16181058,  0.01391721],
        [0.,  0.,  0.,  0.,  0.,  0.],
        [0.,  0.,  4.04714150, -6.34535405,  2.79546509, -0.04801624],
        [0.,  0., -3.94116007, 10.90400303, -6.72931751,  0.41751622],
        [0.,  0.,  2.84194470, -7.54767586,  4.95763672, -0.57428174],
        [0.,  0., -1.61098864,  4.21891584, -2.95010387,  0.47312904],
        [0.,  0.,  1.52360117, -4.32946684,  3.08813015, -0.28226449]
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
    def __init__(self, f, alpha, phi, t_span,  beta=False, phi_t=False, neutral=False):
        ndim, n_state_delays, n_neutral_delays, f, alpha, phi, t_span, beta, phi_t = validade_arguments(
            f, alpha, phi, t_span,  beta=beta, phi_t=phi_t)
        self.t_span = np.array(t_span)
        self.ndim, self.n_state_delays, self.n_neutral_delays = ndim, n_state_delays, n_neutral_delays
        self.f, self.alpha, self.phi, self.t_span = f, alpha, phi, t_span
        self.beta, self.phi_t = beta, phi_t
        self.y_type = np.zeros(self.ndim, dtype=float).dtype
        self.neutral = neutral


class Solution:
    def __init__(self, problem: Problem, discs=[]):
        self.problem = problem
        self.t = [problem.t_span[0]]
        self.y = [np.atleast_1d(problem.phi(problem.t_span[0]))]
        self.etas = [problem.phi]
        self.etas_t = [problem.phi_t]
        if len(discs) == 0:
            discs.append(problem.t_span[0])
        if problem.t_span[0] not in discs:
            discs.append(problem.t_span[0])
        print('discs before', discs)
        self.discs = sorted(discs)
        print('discs after', self.discs)
        # input('daudn')
        self.eta_calls = 0
        self.eta_t_calls = 0
        self.t_next = None

    @property
    def eta(self, ov=False, limit_direction=False):
        def eval(t, ov=ov, limit_direction=limit_direction):
            self.eta_calls += 1
            t = np.atleast_1d(t)  # accept scalar or array
            results = np.empty((len(t), self.problem.ndim), dtype=float)
            for i in range(len(t)):
                idx = bisect_left(self.t, t[i])
                if t[i] < self.t[0]:
                    if limit_direction:
                        # taking a left limit approximation
                        results[i] = self.etas[0](
                            t[i] + limit_direction[i]*np.info(float).eps)
                    else:
                        results[i] = self.etas[0](t[i])
                if t[i] <= self.t[-1]:
                    if limit_direction:
                        # taking a left limit approximation
                        results[i] = self.etas[idx + limit_direction[i]](t[i])
                    else:
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
    def eta_t(self, ov=False):
        def eval(t, ov=ov):
            self.eta_calls += 1
            t = np.atleast_1d(t)  # accept scalar or array
            results = np.empty((len(t), self.problem.ndim), dtype=float)
            for i in range(len(t)):
                idx = bisect_left(self.t, t[i])
                if t[i] <= self.t[-1]:
                    results[i] = self.etas_t[idx](t[i])
                else:
                    if ov:
                        results[i] = self.etas_t[-1](t[i])
                    else:
                        raise ValueError(
                            f"eta isn't defined in {t[i]}, only on {self.t[0], self.t[-1]}")
            return np.squeeze(results)
        return eval

    def update(self, onestep):
        success, step = onestep
        if step.disc != False:
            self.discs.append(step.disc)
            print(f'before accepting {step.disc} and t = {
                  step.t[0] + step.h}')

        if success:  # Step accepted
            # if (self.t[-1] + step.h != step.t[1]):
            # print('sum', self.t[-1] + step.h, 't1', step.t[1])
            self.t.append(step.t[0] + step.h)
            self.y.append(step.y[1])
            self.etas.append(step.new_eta[1])
            self.etas_t.append(step.new_eta_t[1])
            # h = step.h_next  # Use adjusted stepsize from rejection
            return None

        else:
            raise ValueError("Failed")
            return "Failed"


def solve_dde(f, alpha, phi, t_span, method='RK45', neutral=False, beta=None, d_phi=None, discs=[]):
    problem = Problem(f, alpha, phi, t_span, beta=beta,
                      phi_t=d_phi, neutral=neutral)
    solution = Solution(problem, discs=discs)
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
