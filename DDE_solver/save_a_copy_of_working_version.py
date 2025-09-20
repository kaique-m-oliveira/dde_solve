import time
import numbers
from bisect import bisect_left
from dataclasses import dataclass, field
import random
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import lu_factor, lu_solve, norm
from scipy.optimize import root
from scipy.integrate import solve_ivp


@dataclass
class CRKParameters:
    theta1: float = 1 / 3
    TOL: float = 1e-5
    rho: float = 0.9
    omega_min: float = 0.5
    omega_max: float = 1.5


def vectorize_func(func):
    def wrapper(*args, **kwargs):
        # return np.array(func(*args, **kwargs))
        return np.atleast_1d(func(*args, **kwargs))
    return wrapper


def validade_arguments(f, alpha, phi, t_span, d_f, d_alpha, d_phi):
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
        ndelays = 1
    elif isinstance(alpha0, (list, np.ndarray)):
        ndelays = len(alpha0)
    else:
        raise TypeError(f"Unsupported type for alpha(t0, phi(t0)): {alpha0}")

    f = vectorize_func(f)
    alpha = vectorize_func(alpha)
    phi = vectorize_func(phi)

    return ndim, ndelays, f, alpha, phi, t_span


class RungeKutta:
    def __init__(self, problem, solution, h):

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
        self.new_eta = [None, None]
        self.new_eta_t = [None, None]
        self.disc_local_error = None
        self.uni_local_error = None
        self.params = CRKParameters()
        self.overlap = False
        self.test = False
        self.disc = False
        self.ndim = problem.ndim
        self.ndelays = problem.ndelays
        self.fails = 0
        self.stages_calculated = 0
        self.store_times = []
        self.number_of_calls = 0

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
            results = []
            for ti in t:
                if ti <= self.t[0]:
                    results.append(self.solution.eta_t(ti))
                else:
                    results.append(self._hat_eta_0_t(ti))
            return np.squeeze(results)
        return eval

    def is_there_disc(self):
        tn, h, yn = self.t[0], self.h, self.y[0]
        f, eta, alpha = self.problem.f, self.solution.etas[-1], self.problem.alpha
        discs = self.solution.discs
        hn = self.solution.t[-1] - self.solution.t[-2]
        # FIX: maybe not work as well
        if hn <= 1e-15:
            return False
        theta = 1 + h/hn

        def d_zeta(t, disc):
            return alpha(t, eta(t)) - np.full(self.ndelays, disc)

        disc = None
        for disc in discs:
            is_delay = d_zeta(tn, disc) * d_zeta(tn + theta * h, disc) < 0
            # input(f'is delay {is_delay}')
            disc_position = d_zeta(tn, disc) * d_zeta(tn + theta * h, disc) < 0
            if np.any(disc_position):
                new_disc = self.get_disc(disc, disc_position)
                self.disc = new_disc
                return True
        return False

    def get_disc(self, disc, disc_position):
        rho, TOL = self.params.rho, self.params.TOL
        alpha = self.problem.alpha
        alpha_t, alpha_y = self.problem.d_alpha
        eta, eta_t = self.solution.etas[-1], self.solution.etas_t[-1]
        iter, max_iter = 0, 30
        t = self.t[0] + self.h/2

        # def d_zeta(t, disc):
        t_values = []
        for index in np.where(disc_position)[0].tolist():
            def d_zeta(t):
                a = alpha(t, eta(t))[index]
                b = disc
                return a - b

            # FIX: for now, let's use this
            sol = root(d_zeta, t, method='hybr')

            # def d_zeta_t(t, disc):
            #     print(f't {t}')
            #     a = alpha_t(t, eta(t))[index]
            #     print(f'a {a} shape {a.shape}')
            #     b = alpha_y(t, eta(t))[index]
            #     print(f'b {b} shape {b.shape}')
            #     c = eta_t(t)
            #     print(f'c {c} shape {c.shape}')
            #     return a + b * c
            # for i in range(max_iter):
            #     val = abs(d_zeta(t, disc))
            #     if np.any(val < rho*TOL):
            #         break
            #     a = -d_zeta(t, disc)
            # t += -d_zeta(t, disc)/ d_zeta_t(t, disc)

            t_values.append(sol.x)
        return min(t_values)

    def one_step_RK4(self):
        tn, h, yn = self.t[0], self.h, self.y[0]

        f, eta, alpha = self.problem.f, self.eta, self.problem.alpha
        n_stages = self.n_stages["discrete_method"]
        c = self.c[:n_stages]
        A = self.A[:n_stages, :n_stages]
        self.K[0] = f(tn, yn, eta(alpha(tn, yn)))
        self.stages_calculated = 1

        for i in range(1, n_stages):
            ti = tn + c[i] * h
            yi = yn + h * (A[i][0:i] @ self.K[0: i])

            if np.all(alpha(ti, yi) <= np.full(self.ndelays, tn)):
                alpha_i = alpha(ti, yi)
                real_alpha_i = alpha(ti, yi)
                Y_tilde = eta(alpha_i)
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

        print(f'tn = {tn}, h = {
              h}, yn+1 = {self.y[1]}, yn.shape {self.y[1].shape} ')
        # print(f'shape of K {self.K[0].shape}  {self.K[1].shape}  {
        #       self.K[2].shape} {self.K[3].shape}')
        print('__________________________________________________________')
        # input('RK4 stuff')
        # print(
        #     f'tn+1 = {tn + h}, yn+1 = {self.y[1]} real_sol {real_sol(tn + h)}')
        # print(f' ERROR {self.y[1] - real_sol(tn + h)}')
        return True

    def _simplified_Newton(self, alpha_i):

        first_stage = self.stages_calculated
        final_stage = self.n_stages["continuous_ovl_method"]
        total_stages = final_stage - first_stage if final_stage > first_stage else 1

        A = self.A[first_stage:final_stage, first_stage:final_stage]
        c = self.c
        rho, TOL = self.params.rho, self.params.TOL
        f_t, f_y, f_x = self.problem.d_f
        eta_t = self.solution.eta_t
        alpha_t, alpha_y = self.problem.d_alpha
        tn, h, yn = self.t[0], self.h, self.y[0]
        f, eta, alpha = self.problem.f, self.eta, self.problem.alpha
        alpha_n = alpha(tn, yn)
        f_y_n = f_y(tn, yn, eta(alpha_n))
        f_x_n = f_x(tn, yn, eta(alpha_n))
        print('f_y_n', f_y_n, 'shape', f_y_n.shape)
        print('f_x_n', f_x_n, 'shape', f_x_n.shape)
        alpha_y_n = alpha_y(tn, yn)
        pol_order = self.D_ovl.shape[1]

        # WARN: this one is mine
        sum_1 = np.sum(f_x_n * self.eeta_t(alpha_n) * alpha_y_n)

        # theta = np.squeeze((alpha_n - tn) / h)
        theta = np.squeeze((alpha_i - tn) / h)
        if self.ndelays == 1:
            theta = theta ** np.arange(pol_order)
            print('first_stage', first_stage)
            print('final_stage', final_stage)
            D_ovl = self.D_ovl[first_stage:final_stage, :]
            D = D_ovl @ theta
            B = np.tile(D[:, None], (1, total_stages))
            print('B', B)
            sum_2 = np.kron(B, f_x_n)

        else:
            sum_2 = 0
            print('t0', self.t[0])
            print('alpha_n', alpha_n)
            for i in range(self.ndelays):
                print('i', i)
                if alpha_i[i] <= self.t[-1]:
                    B = np.zeros((total_stages, total_stages), dtype=yn.dtype)
                    print('making this mf zero')
                else:
                    theta_i = theta[i] ** np.arange(pol_order)
                    D_ovl = self.D_ovl[first_stage:final_stage, :]
                    D = D_ovl @ theta_i
                    print('D', D)
                    B = np.tile(D[:, None], (1, total_stages))
                    print('B', B)
                    print('this was supposed not to be zero')

                print('f_x_n[i]', f_x_n[i], 'shape', f_x_n[i].shape)
                kron = np.kron(B, f_x_n[i])
                print('kron', kron, 'shape', kron.shape)
                sum_2 += kron

        # I = np.eye(total_stages, dtype=yn.dtype)
        # I = np.eye(total_stages * self.ndim, dtype=yn.dtype)
        I = np.kron(np.eye(total_stages, dtype=yn.dtype),
                    np.eye(self.ndim, dtype=yn.dtype))

        # print('I', I, 'shape', I.shape)
        first = - h * np.kron(A, f_y_n + sum_1)
        # print('A', A, 'shape', A.shape)
        # print('first', first, 'shape', first.shape)
        #
        second = - h * sum_2
        print('K', self.K)
        print('sum_2', sum_2, 'shape', sum_2.shape)
        print('second', second, 'shape', second.shape)
        J = I + first + second

        # print('t0', self.t[0], 'h', self.h)
        # print('K', self.K)
        # print('J', J, 'shape', J.shape)

        lu, piv = lu_factor(J)

        def F(K):
            F = np.zeros((total_stages, self.ndim), dtype=float)
            for i in range(total_stages):
                ti = tn + c[first_stage + i] * h
                yi = yn + c[first_stage + i] * h * K[i]
                Y_tilde = self.eeta(alpha(ti, yi))
                F[i] = K[i] - f(ti, yi, Y_tilde)
            return F.ravel()

        max_iter, iter = 100, 0
        err = 100

        # first iteration
        inside_K = self.K[first_stage:final_stage]
        FK = F(inside_K)
        print('FK', FK, 'shape', FK.shape)
        # input(f'lu {lu} shape {lu.shape}')
        diff_old = lu_solve((lu, piv), - FK).reshape(total_stages, self.ndim)
        self.K[first_stage:final_stage] += diff_old
        iter += 1

        inside_K = self.K[first_stage:final_stage]
        FK = F(inside_K)
        diff_new = lu_solve((lu, piv), - FK).reshape(total_stages, self.ndim)
        self.K[first_stage:final_stage] += diff_new
        iter += 1
        err = abs((np.linalg.norm(diff_new)**2) /
                  (np.linalg.norm(diff_old) - np.linalg.norm(diff_new)))

        while err >= rho * TOL and iter <= max_iter:
            # Método de Newton usando recomposição LU
            diff_old = diff_new

            inside_K = self.K[first_stage:final_stage]
            FK = F(inside_K)
            diff_new = lu_solve(
                (lu, piv), - FK).reshape(total_stages, self.ndim)
            self.K[first_stage:final_stage] += diff_new
            iter += 1
            err = abs((np.linalg.norm(diff_new)**2) /
                      (np.linalg.norm(diff_old) - np.linalg.norm(diff_new)))

        if iter > max_iter:
            # input(f'falhou com erro {err} e {iter} iterações')
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
        pol_order = self.D_ovl.shape[1]
        theta = np.array([n*theta**(n-1) for n in range(pol_order)])
        K = self.K[0:self.n_stages["continuous_ovl_method"]]
        bs = (self.D_ovl @ theta).squeeze()
        eta0 = yn + h * bs @ K
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
        # print('shape theta', theta.shape)
        # print('shape D', self.D.shape)
        # print('shape self.D @ theta', (self.D @ theta).shape)
        # print('shape K', K.shape)
        # input('bro')
        bs = (self.D @ theta).squeeze()
        eta0 = yn + h * bs @ K
        return eta0

    def error_est_method(self):
        f, alpha = self.problem.f,  self.problem.alpha
        if self.n_stages["discrete_err_est_method"] - self.stages_calculated <= 0:
            return
        else:
            for i in range(self.stages_calculated, self.n_stages["discrete_err_est_method"]):
                ti = self.t[0] + self.c[i] * self.h
                yi = self.y[0] + self.h * (self.A[i][0:i] @ self.K[0: i])
                alpha_i = alpha(ti, yi)
                Y_tilde = self.eeta(alpha_i)
                self.K[i] = f(ti, yi, Y_tilde)
            self.stages_calculated = self.n_stages["discrete_err_est_method"]

        K = self.K[0:self.n_stages["discrete_err_est_method"]]
        self.y_tilde = self.y[0] + self.h * (self.b_err @ K)

    def disc_local_error_satistied(self):
        self.disc_local_error = (
            np.linalg.norm(self.y_tilde - self.y[1]) / self.h
        )  # eq 7.3.4

        if self.disc_local_error <= self.params.TOL:
            return True
        else:
            self.h = min(1, self. h * (
                max(
                    self.params.omega_min,
                    min(
                        self.params.omega_max,
                        self.params.rho
                        * (self.params.TOL / self.disc_local_error) ** (1 / 4),
                    ),
                )
            ))

            self.t = [self.t[0], self.t[0] + self.h]
            return False

    def uni_local_error_satistied(self):

        tn, h = self.t[0], self.h
        self.uni_local_error = (
            h * np.linalg.norm(self.new_eta[0](tn + (1/2) * h) - self.new_eta[1](tn + (1/2)*h)))
        # eq 7.3.4

        if self.uni_local_error <= self.params.TOL:
            return True
        else:
            self.h = min(1, self.h * (
                max(
                    self.params.omega_min, self.params.rho *
                    (self.params.TOL / self.uni_local_error) ** (1 / 5)
                )
            ))
            self.t = [self.t[0], self.t[0] + self.h]
            return False

    def try_step_CRK(self):
        success = self.one_step_RK4()
        if not success:
            self.h = self.h/2
            return False

        self.build_eta_1()
        self.new_eta[1] = self._eta_1
        self.build_eta_0()
        self.new_eta[0] = self._eta_0
        self.new_eta_t = [self._eta_1_t, self._eta_1_t]
        self.error_est_method()
        uni_local_disc_satistied = self.uni_local_error_satistied()

        if not uni_local_disc_satistied:
            print(f'failed uniform step at t = {
                  self.t[0] + self.h} with h = {self.h}')
            return False

        local_disc_satisfied = self.disc_local_error_satistied()

        if not local_disc_satisfied:
            print(f'failed discrete step at t = {
                  self.t[0] + self.h} with h = {self.h}')
            return False

        # print(f'successfull step with h = {self.h}')
        # Handling divide by zero case
        if self.disc_local_error < 1e-14 or self.uni_local_error < 1e-14:
            self.h_next = min(1, self.params.omega_max * self.h)
        else:
            self.h_next = min(1, self.h * max(
                self.params.omega_min,
                min(
                    self.params.omega_max,
                    self.params.rho * (self.params.TOL /
                                       self.disc_local_error)**(1/4),
                    self.params.rho * (self.params.TOL /
                                       self.uni_local_error)**(1/5)
                )
            ))

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
                    if new_h < self.h:
                        print('disc_found', disc_found)
                        if new_h not in self.solution.discs:
                            self.solution.discs.append(disc_found)
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


class Problem:
    def __init__(self, f, alpha, phi, t_span, d_f=None, d_alpha=None, d_phi=None):
        ndim, ndelays, f, alpha, phi, t_span = validade_arguments(
            f, alpha, phi, t_span, d_f, d_alpha, d_phi)
        self.t_span = np.array(t_span)
        self.ndim, self.ndelays, self.f, self.alpha, self.phi, self.t_span = ndim, ndelays, f, alpha, phi, t_span
        self.y_type = np.zeros(self.ndim, dtype=float).dtype
        self.d_alpha = self.get_d_alpha()
        self.d_f = self.get_d_f()
        self.d_phi = self.get_d_phi()

    def get_d_phi(self):
        phi = self.phi
        h = 1e-15

        def d_phi(t):
            return (phi(t) - phi(t - h))/h

        return d_phi

    def get_d_f(self):
        alpha = self.alpha
        f = self.f
        ndim = self.ndim
        ndelays = self.ndelays
        d_alpha = np.empty(ndelays, dtype=self.y_type)
        h = 1e-15

        def unit_vec(j): return np.array(
            [1 if i == j else 0 for i in range(ndim)])

        def f_t(t, y, x):
            return (f(t, y, x) - f(t - h, y, x))/h

        def f_y(t, y, x):
            val = np.zeros((self.ndim, self.ndim), dtype=float)
            print('val before', val, 'shape', val.shape)
            for j in range(ndim):
                val_j = (f(t, y, x) - f(t, y - h*unit_vec(j), x))/h
                print('val_j', val_j, 'shape', val_j.shape)
                val[j] = val_j
            print('val', val, 'shape', val.shape)
            # input('f_y')
            return np.atleast_1d(val)

        def x_add(x, h, j):
            x[j] -= h
            return x

        if ndelays == 1:
            def f_x(t, y, x):
                # delays = np.empty((ndelays, ndelays * ndim), dtype=y.dtype)
                delays = np.zeros((ndelays, ndim, ndim), dtype=y.dtype)
                print('delays', delays)
                for i in range(ndelays):
                    # val = np.zeros(self.ndim, dtype=float)
                    val = np.zeros((self.ndim, self.ndim), dtype=float)
                    for j in range(ndim):
                        val[j] = (f(t, y, x) - f(t, y, x - h*unit_vec(j)))/h
                    print('val', val)
                    delays[i] = np.atleast_1d(val)
                return np.squeeze(delays)

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
                return np.squeeze(delays)

        return f_t, f_y, f_x

    def get_d_alpha(self):
        alpha = self.alpha
        ndim = self.ndim
        d_alpha = [None, None]
        h = 1e-15

        def unit_vec(j): return np.array(
            [1 if i == j else 0 for i in range(ndim)])

        def alpha_t(t, y):
            return (alpha(t, y) - alpha(t - h, y))/h

        if self.ndelays == 1:
            def alpha_y(t, y):
                val = np.zeros(self.ndim, dtype=float)
                for j in range(ndim):
                    val[j] = (alpha(t, y) - alpha(t, y - h*unit_vec(j)))/h
                return np.atleast_1d(val)
        else:
            def alpha_y(t, y):
                delays = np.empty((self.ndelays, self.ndim), dtype=y.dtype)
                for i in range(self.ndelays):
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
        def eval(t, epsilon=1e-15):
            self.eta_calls += 1
            t = np.atleast_1d(t)  # accept scalar or array
            results = []
            for ti in t:
                idx = bisect_left(self.t, ti)
                if ti <= self.t[-1]:
                    results.append(self.etas_t[idx](ti))
                else:
                    raise ValueError(
                        f"eta_t isn't defined in {ti}, only on {self.t[0], self.t[-1]}")
            return np.squeeze(results)
        return eval

    # FIX: this shit is not working at all
    def build_eta(self):
        if self.problem.ndelays == 1:
            def eta(t):
                idx = bisect_left(self.t, t)
                if t <= self.t[-1]:
                    return self.etas[idx](t)
                else:
                    raise ValueError(
                        f"eta isn't defined in {t}, only on {self.t[0], self.t[-1]}")
        else:
            def eta(t):
                t = np.atleast_1d(t)
                results = np.empty((len(t), self.problem.ndim), dtype=float)
                for i in range(len(t)):
                    idx = bisect_left(self.t, t[i])
                    if t[i] <= self.t[-1]:
                        results[i] = self.etas[idx](t[i])
                    else:
                        raise ValueError(
                            f"eta isn't defined in {t[i]}, only on {self.t[0], self.t[-1]}")
                return np.squeeze(results)
        return eta

    def update(self, onestep):
        success, step = onestep
        if step.disc != False:
            self.discs.append(step.disc)

        if success:  # Step accepted
            if (self.t[-1] + step.h != step.t[1]):
                print('sum', self.t[-1] + step.h, 't1', step.t[1])
            self.t.append(step.t[0] + step.h)
            self.y.append(step.y[1])
            self.etas.append(step.new_eta[1])
            self.etas_t.append(step.new_eta_t[1])
            # h = step.h_next  # Use adjusted stepsize from rejection
            return None

        else:
            raise ValueError("Failed")
            return "Failed"


def solve_dde(f, alpha, phi, t_span, d_f=None, d_alpha=None, d_phi=None):
    problem = Problem(f, alpha, phi, t_span, d_f, d_alpha, d_phi)
    solution = Solution(problem)
    params = CRKParameters()
    t, tf = problem.t_span

    h = (params.TOL ** (1 / 4)) * 0.1  # Initial stepsize
    print("-" * 80)
    print("Initial h:", h)
    print("-" * 80)

    first_step = RK4HHL(problem, solution, h)
    status = solution.update(first_step.first_step_CRK())
    if status != None:
        raise ValueError(status)
    h = first_step.h_next
    t = solution.t[-1]

    times = []
    calls = 0
    while t < tf:
        h = min(h, tf - t)
        onestep = RK4HHL(problem, solution, h)
        status = solution.update(onestep.one_step_CRK())
        calls += onestep.number_of_calls
        # if onestep.store_times != []:
        #     times.extend(onestep.store_times)
        # print(f'eta({t}) {solution.eta(t)}')
        if status != None:
            raise ValueError(status)
        h = onestep.h_next
        t = solution.t[-1]

    # print('real calls', calls)
    # print('total sum', sum(times))
    # print('total accesses', len(times))
    # print(f'final avg time {sum(times)/len(times)}')
    return solution
