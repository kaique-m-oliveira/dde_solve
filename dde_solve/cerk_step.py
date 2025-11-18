import numpy as np

from DDE_solve.tools import *


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
        self.first_step = False

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
                    results[i] = self.eta_t(t[i])
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

    def reset_step(self):
        total_stages = self.A.shape[0]
        self.h_next = None
        self.y_tilde = None
        self.K = np.zeros((total_stages, self.n), dtype=float)
        self.new_eta = [None, None]
        self.new_eta_t = [None, None]
        self.disc_local_error = None
        self.uni_local_error = None
        self.overlap = False
        self.test = False
        self.disc = None  # either False or a pair (disc_old, disc_new)
        self.fails = 0
        self.stages_calculated = 0
        self.first_eta = True
        self.disc_position = False
        self.disc_beta_positions = False
        self.disc_interval = None
        self.breaking_step = False
        self.disc_flag = False

    def is_there_disc(self):
        tn, h = self.t[0], self.h
        eta, alpha = self.solution.etas[-1], self.problem.alpha
        if self.neutral:
            beta = self.problem.beta
        discs = self.solution.discs

        if h <= 1e-15:
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

        if d_zeta(a)*d_zeta(b) < 0:
            return True
        else:
            return False

    def one_step_RK4(self, eta_ov=None, eta_t_ov=None):

        total_stages = self.A.shape[0]
        self.K = np.zeros((total_stages, self.n), dtype=float)
        tn, h, yn = self.t[0], self.h, self.y[0]
        eta = self.eta
        f, alpha = self.problem.f, self.problem.alpha
        n_stages = self.n_stages["discrete_method"]
        c = self.c[:n_stages]
        A = self.A[:n_stages, :n_stages]
        for i in range(0, n_stages):
            ti = tn + c[i] * h
            yi = yn + h * (A[i][0:i] @ self.K[0: i])

            alpha_i = alpha(ti, yi)
            if np.all(alpha_i <= np.full(self.ndelays, tn)):
                Y_tilde = eta(alpha_i)

            elif eta_ov is not None:
                Y_tilde = eta_ov(alpha_i)

            else:  
                self.overlap = True
                success = self.fixed_point()
                if not success:
                    return False
                break

            if not self.neutral:
                self.K[i] = f(ti, yi, Y_tilde)
                self.solution.feval += 1
                self.stages_calculated = i + 1

            else:
                beta_i = self.problem.beta(ti, yi)

                if np.all(beta_i <= np.full(self.ndelays, tn)):
                    Z_tilde = self.eta_t(beta_i)

                elif eta_t_ov is not None:
                    Z_tilde = eta_t_ov(beta_i)

                else:  # this would be the overlapping case
                    self.overlap = True
                    success = self.fixed_point()
                    if not success:
                        return False
                    break

                self.K[i] = f(ti, yi, Y_tilde, Z_tilde)
                self.solution.feval += 1
                self.stages_calculated = i + 1

        self.y[1] = yn + h * (self.b @ self.K[0:n_stages])
        self.stages_calculated = n_stages

        # safety condition
        if np.isnan(self.y[1]).any():
            return False

        return True


    def fixed_point(self):
        max_iter = 12
        sc = self.Atol + np.abs(self.y[0]) * self.Rtol

        self.K_prev = self.K[0:self.n_stages["discrete_method"]].copy()
        if self.first_step:
            self.first_eta = True
            successfull_interpolant = self.special_interpolant()
            if not successfull_interpolant:
                return False

            self.first_eta = False

        self.one_step_RK4(eta_ov=self.eeta, eta_t_ov=self.eeta_t)
        self.first_eta = False

        for i in range(max_iter):
            K_new = self.K[0:self.n_stages["discrete_method"]].copy()

            diff = np.abs(K_new - self.K_prev) / sc
            err_stage = np.linalg.norm(diff, axis=1) / np.sqrt(self.ndim)

            if np.max(err_stage) <= 1:
                return True

            # prepare next iteration: freeze eta from previous K
            self.K_prev = K_new.copy()
            self.one_step_RK4(eta_ov=self.eeta, eta_t_ov=self.eeta_t)

        return False



    def special_interpolant(self, eta_ov=None, eta_t_ov=None):
        """Special interpolant used for first step, described on the paper from Enright and Hayashi"""

        total_stages = self.A.shape[0]
        tn, h, yn = self.t[0], self.h, self.y[0]
        eta = self.eta
        f, alpha = self.problem.f, self.problem.alpha
        n_stages = self.n_stages["discrete_method"]
        c = self.c[:n_stages]
        A = self.A[:n_stages, :n_stages]


        # Stage 1 (Y1 = z_{n-1}(tn) and k1 uses history)
        alpha1 = alpha(tn, yn)
        Y_tilde1 = eta(alpha1)  # vectorized
        if not self.neutral:
            self.K_prev[0] = f(tn, yn, Y_tilde1)
            self.solution.feval += 1
        else:
            beta1 = self.problem.beta(tn, yn)
            Z_tilde1 = self.solution.eta_t(beta1)
            self.K_prev[0] = f(tn, yn, Y_tilde1, Z_tilde1)
            self.solution.feval += 1

        # Stage 2: linear interpolant between (tn, yn) and (tn + c2*h, y2)
        t2 = tn + c[1] * h
        y2 = yn + h * A[1, 0] * self.K_prev[0]
        alpha2 = alpha(t2, y2)               # may be scalar or array
        eta2, eta2_t = linear_interpolant(tn, h, c[1], yn, y2)

        #TODO: Need to wrap all these safety precations somewhere else 

        # vectorized selection: if alpha2 <= tn we query history eta, else eta2
        alpha2_arr = np.atleast_1d(alpha2)
        mask = alpha2_arr <= tn

        # prepare output array shape (m, ndim) or (ndim,) for scalar
        if mask.all():
            Y_tilde_2 = eta(alpha2_arr)
        elif (~mask).all():
            Y_tilde_2 = eta2(alpha2_arr)
        else:
            # mixed: evaluate both and assemble
            Y_full = np.empty((len(alpha2_arr), self.ndim))
            if mask.any():
                Y_full[mask] = np.atleast_2d(eta(alpha2_arr[mask]))
            if (~mask).any():
                Y_full[~mask] = np.atleast_2d(eta2(alpha2_arr[~mask]))
            Y_tilde_2 = np.squeeze(Y_full)

        if not self.neutral:
            self.K_prev[1] = f(t2, y2, Y_tilde_2)
            self.solution.feval += 1
        else:
            beta2 = self.problem.beta(t2, y2)
            beta2_arr = np.atleast_1d(beta2)
            mask_b = beta2_arr <= tn
            if mask_b.all():
                Z_tilde_2 = self.solution.eta_t(beta2_arr)
            elif (~mask_b).all():
                Z_tilde_2 = eta2_t(beta2_arr)
            else:
                Z_full = np.empty((len(beta2_arr), self.ndim))
                if mask_b.any():
                    Z_full[mask_b] = np.atleast_2d(self.solution.eta_t(beta2_arr[mask_b]))
                if (~mask_b).any():
                    Z_full[~mask_b] = np.atleast_2d(eta2_t(beta2_arr[~mask_b]))
                Z_tilde_2 = np.squeeze(Z_full)

            self.K_prev[1] = f(t2, y2, Y_tilde_2, Z_tilde_2)
            self.solution.feval += 1

        # Remaining stages using the quadratic interpolation in the overlapping case
        for i in range(2, n_stages):
            ti = tn + c[i] * h
            yi = yn + h * (A[i, :i] @ self.K_prev[:i])

            alpha_i = alpha(ti, yi)  
            eta_i, eta_i_t = quadratic_interpolant(tn, h, c[i], yn, self.K_prev[0], yi)

            alpha_arr = np.atleast_1d(alpha_i)
            mask_a = alpha_arr <= tn
            if mask_a.all():
                Y_tilde = eta(alpha_arr)
            elif (~mask_a).all():
                Y_tilde = eta_i(alpha_arr)
            else:
                Y_full = np.empty((len(alpha_arr), self.ndim))
                if mask_a.any():
                    Y_full[mask_a] = np.atleast_2d(eta(alpha_arr[mask_a]))
                if (~mask_a).any():
                    Y_full[~mask_a] = np.atleast_2d(eta_i(alpha_arr[~mask_a]))
                Y_tilde = np.squeeze(Y_full)

            if not self.neutral:
                self.K_prev[i] = f(ti, yi, Y_tilde)
                self.solution.feval += 1
            else:
                beta_i = self.problem.beta(ti, yi)
                beta_arr = np.atleast_1d(beta_i)
                mask_b = beta_arr <= tn
                if mask_b.all():
                    Z_tilde = self.solution.eta_t(beta_arr)
                elif (~mask_b).all():
                    Z_tilde = eta_i_t(beta_arr)
                else:
                    Z_full = np.empty((len(beta_arr), self.ndim))
                    if mask_b.any():
                        Z_full[mask_b] = np.atleast_2d(self.solution.eta_t(beta_arr[mask_b]))
                    if (~mask_b).any():
                        Z_full[~mask_b] = np.atleast_2d(eta_i_t(beta_arr[~mask_b]))
                    Z_tilde = np.squeeze(Z_full)

                self.K_prev[i] = f(ti, yi, Y_tilde, Z_tilde)
                self.solution.feval += 1

            diff = np.linalg.norm(self.K_prev[i] - self.K_prev[i-1], ord=np.inf)
            if diff > 1e6:
                return False

            self.stages_calculated = i + 1

        return True



    #TODO: Make a general purpose stage builder with switching conditions 

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
                    self.solution.feval += 1
                else:
                    self.K[i] = f(ti, yi, Y_tilde)
                    self.solution.feval += 1
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
        K = self.K_prev[0:self.n_stages["continuous_ovl_method"]]
        bs = (self.D_ovl @ theta).squeeze()
        eta0 = yn + h * bs @ K
        return eta0

    def _hat_eta_0_t(self, theta):
        tn, h, yn = self.t[0], self.h, self.y[0]
        tt = theta
        theta = (theta - tn) / h
        pol_order = self.D_ovl.shape[1]
        n = np.arange(pol_order)
        theta = np.array([n*theta**(n-1) if n != 0 else 0 for n in range(pol_order)])
        K = self.K_prev[0:self.n_stages["continuous_ovl_method"]]
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
                    Z_tilde = self.eeta_t(beta_i)
                    self.K[i] = f(ti, yi, Y_tilde, Z_tilde)
                    self.solution.feval += 1
                else:
                    self.K[i] = f(ti, yi, Y_tilde)
                    self.solution.feval += 1
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
        tt = theta
        theta = (theta - tn) / h
        pol_order = self.D.shape[1]
        n = np.arange(pol_order)
        theta = np.array([n*theta**(n-1) if n != 0 else 0 for n in range(pol_order)])
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
                    Z_tilde = self.eeta_t(beta_i)
                    self.K[i] = f(ti, yi, Y_tilde, Z_tilde)
                    self.solution.feval += 1
                else:
                    self.K[i] = f(ti, yi, Y_tilde)
                    self.solution.feval += 1
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


    def uniform_disc_satistied(self):

        tn, h = self.t[0], self.h
        sc = self.Atol + np.abs(self.y[0])*self.Rtol
        n_cont = self.n_stages["continuous_err_est_method"]   # or len(self.c) if appropriate
        c_points = self.c[:n_cont]
        val1 = np.vstack([self.new_eta[0](tn + ci*h) for ci in c_points])
        val2 = np.vstack([self.new_eta[1](tn + ci*h) for ci in c_points])
        diffs = np.abs(val1 - val2) / sc  # shape (n_cont, ndim)
        errs_per_sample = np.linalg.norm(diffs, axis=1) / np.sqrt(self.ndim)  # shape (n_cont,)
        self.uni_local_error = np.max(errs_per_sample)
        if self.uni_local_error <= 1:
            return True
        else:
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
        self.error_est_method()

        # Safety condition
        if np.isnan(self.K).any() or np.isinf(self.K).any():
            self.h = self.h/2
            self.h_next = self.h
            return False

        discrete_disc_satisfied = self.discrete_disc_satistied()

        uniform_disc_satistied = self.uniform_disc_satistied()

        facmax = 1.2
        facmin = 0.5
        fac = 0.9
        err1 = self.disc_local_error if self.disc_local_error >= 1e-15 else 1e-15
        err2 = self.uni_local_error if self.uni_local_error >= 1e-15 else 1e-15
        pp = min(self.order["discrete_method"],
                 self.order["discrete_err_est_method"])
        qq = min(self.order["continuous_method"],
                 self.order["continuous_err_est_method"])

        self.t[1] = self.t[0] + self.h

        self.h_next = self.h * \
            min(facmax, max(facmin, fac*min((1/err1) **
                (1/(pp + 1)), (1/err2)**(1/(qq + 1)))))

        self.h_next = min(self.h_next, 0.3)

        if not discrete_disc_satisfied or not uniform_disc_satistied:
            self.h = self.h_next
            return False

        return True

    def investigate_disc(self):
        true_disc = self.validade_disc()

        if not true_disc:
            self.disc = None
            return

        self.disc = self.disc_interval[1]
        self.h = self.disc - self.t[0]

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
            return delay(t, eta(t)) - disc  

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

        #TODO: Make this bit arithmatic more inteligible

        for mask in range(1 << N):      # loop over 0..2^k-1
            limit_direction = alpha_limits.copy()
            for j in range(N):
                if (mask >> j) & 1:     # check j-th bit
                    limit_direction[idx[j]] = -1

            t1 = self.t[1]
            y1 = self.y[1]

            alpha1 = alpha(t1, y1)
            alpha1 = [alpha1[i] if alpha_discs[i] is None else alpha_discs[i]
                      for i in range(len(alpha1))]


            if not self.neutral:
                limit_directions.append(limit_direction)
                y_lim = y1 + eps * \
                    f(t1, y1, eta(alpha1, limit_direction=limit_direction))
                continued = -1*limit_direction * \
                    (alpha(t1 + eps, y_lim) - old_disc) < 0
                mask = np.array(alpha_limits.astype(bool))

                continued = continued[mask]
                continuation.append(continued)

            else:
                eta_t = self.solution.eta_t
                limit_directions.append(limit_direction)
                y_lim = y1 + eps * \
                    f(t1, y1, eta(alpha1, limit_direction=limit_direction), eta_t(alpha1 + limit_direction*10**-16, limit_direction=limit_direction) )

                continued = -1*limit_direction * \
                    (alpha(t1 + eps, y_lim) - old_disc) < 0
                mask = np.array(alpha_limits.astype(bool))

                continued = continued[mask]
                continuation.append(continued)

        if not np.any(np.all(continuation, axis=1)):
            return "terminated"

        possible_branches = np.all(continuation, axis=1)
        if sum(possible_branches) == 1:
            pos = np.where(possible_branches)[0][0]
            self.limit_direction = limit_directions[pos]
            return "one branch"
        else:
            pos = np.where(possible_branches)[0].tolist()
            self.limit_directions = np.array(limit_directions)[pos]
            return "branches"

    def one_step_CRK(self, max_iter=20):
        iter = 0
        while self.h >= 10**-12 and iter <= max_iter:
            success = self.try_step_CRK()
            self.solution.steps += 1 
            if success:
                return True, self
            self.reset_step()
            self.solution.fails += 1

            disc_found = self.is_there_disc()
            if disc_found:
                h = self.disc_interval[0] - self.t[0]

                if h >= 1e-14:
                    self.h = h
                    success = self.try_step_CRK()
                    self.solution.steps += 1 
                    if success:
                        self.investigate_disc()
                        return True, self
                    self.solution.fails += 1
                    self.reset_step()
            iter += 1

        return False, 0

    def first_step_investigate_branch(self):

        if self.solution.breaking_discs:
            self.alpha_discs = np.full(self.problem.n_state_delays, None)
            alpha0 = self.problem.alpha(self.t[0], self.y[0])

            for i in range(self.problem.n_state_delays):
                if self.ndim == 1:
                    if float(alpha0[i]) in self.solution.breaking_discs:
                        self.alpha_discs[i] = float(alpha0[i])
                else:
                    for dim in range(self.ndim):
                        if float(alpha0[i][dim]) in self.solution.breaking_discs:
                            self.alpha_discs[i] = float(alpha0[i][dim])

            if np.any(self.alpha_discs):
                self.disc = self.t[0]
                state = self.investigate_branches()
                return state

        return None
