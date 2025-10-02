# WARN: ALL THE EXTRA STUFF FROM NEWTON
from rkh_refactor import *


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

        # FIX: collapsing into scalar, WRONG
        sum_1 = np.sum(f_x_n * self.eeta_t(alpha_n) * alpha_y_n)

        sum_1t = 0
        if self.neutral:
            # FIX: collapsing into scalar, WRONG
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

        while err >= rho * TOL and iter <= max_iter:
            # while np.any(F(inside_K) >= rho * TOL) and iter <= max_iter:

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

        # print('Ks', self.K)
        # print('final K', self.K[first_stage:final_stage])
        if iter > max_iter:

            # print('err', err)
            # print('TOL', TOL, 'rho', rho, rho*TOL)
            # y1 = yn + h * (self.b @ self.K[0:4])
            # print(f'tn = {tn}, h = {
                #       h}, yn+1 = {y1}, yn.shape {y1.shape} ')
            # print(f'yn+1 = {y1} real_sol = {real_sol(tn + h)
                                              #                                        } diff={y1 - real_sol(tn + h)}')
            print('Newton falhou')
            return False
        return True



#Class problem with a bunch of extra shit
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
