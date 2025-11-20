import numpy as np

# from dde_solve import *
from dde_solve import *


def A1(method, Tol):
    def f(t, y, x):
        return 0.2*x/(1+x**10) - 0.1*y
    def phi(t):
        return 0.5
    def alpha(t, y):
        return t - 14
    t_span = [0, 500]
    solution = solve_dde(t_span, f, alpha, phi, method = method, Atol=Tol, Rtol=Tol)
    real_solution = None
    return solution, real_solution

def A2(method, Tol):
    def f(t, y, x):
        y1, y2 = y
        x1, x2 = x
        dy1 = 1.1/(1 + np.sqrt(10)*x1**(5/4)) - 10*y1/(1 + 40*y2)
        dy2 =100*y1/(1+40*y2) - 2.43*y2
        return dy1, dy2
    def phi(t):
        return [1.05767027/3, 1.030713491/3]
    def alpha(t, y):
        return t - 20
    t_span = [0, 100]
    solution = solve_dde(t_span, f, alpha, phi, method = method, Atol=Tol, Rtol=Tol)
    real_solution = None
    return solution, real_solution

def B1(method, Tol):
    def f(t, y, x):
        return 1 - x
    def phi(t):
        return np.log(t)
    def alpha(t, y):
        return np.exp(1 - 1/t)
    t_span = [0.1, 10]
    solution = solve_dde(t_span, f, alpha, phi, method = method, Atol=Tol, Rtol=Tol)
    real_solution = lambda t: np.log(t)
    return solution, real_solution

def B2(method, Tol):
    def f(t, y, x):
        fx = 1.0 if x < 0 else -1.0
        return fx - y
    def phi(t):
        return 1.0
    def alpha(t, y):
        return t / 2.0
    t_span = [0, 2*np.log(66)]
    solution = solve_dde(t_span, f, alpha, phi, method = method, Atol=Tol, Rtol=Tol)
    def real_solution(t):
        if 0 <= t <= 2*np.log(2):
            return 2*np.exp(-t) - 1
        elif 2*np.log(2) < t <= 2*np.log(6):
            return 1 - 6*np.exp(-t)
        elif 2*np.log(6) < t <= 2*np.log(66):
            return 66*np.exp(-t) - 1
    return solution, real_solution

def C1(method, Tol):
    def f(t, y, x):
        return -2 * x * (1 - y**2)
    def phi(t):
        return 0.5
    def alpha(t, y):
        return t - 1 - abs(y)
    t_span = [0, 30]
    solution = solve_dde(t_span, f, alpha, phi, method = method, Atol=Tol, Rtol=Tol)
    real_solution = None
    return solution, real_solution



def C2(method, Tol):
    def f(t, y, x):
        y1, y2 = y
        x1, _ = x  # x = [y1(t - y2(t)), y2(t - y2(t))], but only x1 is used
        dy1 = -2 * x1
        dy2 = (abs(x1) - abs(y1)) / (1 + abs(x1))
        return [dy1, dy2]
    def phi(t):
        return [1.0, 0.5]
    def alpha(t, y):
        y1, y2 = y
        return t - y2
    t_span = [0, 40]
    solution = solve_dde(t_span, f, alpha, phi, method = method, Atol=Tol, Rtol=Tol)
    real_solution = None
    return solution, real_solution


def C3(method, Tol):
    s0_hat = 0.0031
    T1 = 6.0
    gamma = 0.001
    Q = 0.0275
    k = 2.8
    a = 6570
    K = 0.0382
    r = 6.96
    def f_nl(y1):
        return a / (1 + K * y1**r)
    def f(t, y, x):
        y1, y2, y3 = y
        x1, x2 = x
        x11, x12, x13 = x1
        x21, x22, x23 = x2

        dy1 = s0_hat * x12 - gamma * y1 - Q
        dy2 = f_nl(y1) - k * y2
        dy3 = 1 - (Q * np.exp(gamma * y3)) / (s0_hat * x22)
        return [dy1, dy2, dy3]
    def phi(t):
        if t <= -T1:
            phi2 = 9.5
        elif -T1 < t <= 0:
            phi2 = 10.0
        else:
            phi2 = np.nan  # outside history domain
        return [3.325, phi2, 120.0]
    def alpha(t, y):
        y1, y2, y3 = y
        return [t - T1, t - T1 - y3]
    t_span = [0, 300]
    solution = solve_dde(t_span, f, alpha, phi, method = method, Atol=Tol, Rtol=Tol)
    real_solution = None
    return solution, real_solution


def C4(method, Tol):
    s0_hat = 0.00372
    T1 = 3
    gamma = 0.1
    Q = 0.00178
    k = 6.65
    a = 15600
    K = 0.0382
    r = 6.96
    def f_nl(y1):
        return a / (1 + K * y1**r)
    def f(t, y, x):
        y1, y2, y3 = y
        x1, x2 = x
        x11, x12, x13 = x1
        x21, x22, x23 = x2
        dy1 = s0_hat * x12 - gamma * y1 - Q
        dy2 = f_nl(y1) - k * y2
        dy3 = 1 - (Q * np.exp(gamma * y3)) / (s0_hat * x22)
        return [dy1, dy2, dy3]
    def phi(t):
        return [3.5, 10, 50]
    def alpha(t, y):
        y1, y2, y3 = y
        return [t - T1, t - T1 - y3]
    t_span = [0, 100]
    solution = solve_dde(t_span, f, alpha, phi, method = method, Atol=Tol, Rtol=Tol)
    real_solution = None
    return solution, real_solution


def D1(method, Tol):
    def f(t, y, x):
        y1, y2 = y
        x1, x2 = x
        dy1 = y2
        dy2 = -x2*(y2**2)*np.exp(1 - y2)
        return [dy1, dy2]
    def phi(t):
        return [np.log(t), 1/t]
    def alpha(t, y):
        y1, y2 = y
        return np.exp(1 - y2)
    t_span = [0.1, 5]
    solution = solve_dde(t_span, f, alpha, phi, method = method, Atol=Tol, Rtol=Tol)
    def real_solution(t):
        return np.array([np.log(t), 1/t])
    return solution, real_solution


def D2(method, Tol):
    r1 = 0.02
    r2 = 0.005
    alpha_param = 3.0
    delta = 0.01
    def f(t, y, x):
        y1, y2, y3, y4 = y
        xd1, xd2, xd3, xd4 = x  
        dy1 = -r1 * y1 * y2 + r2 * y3
        dy2 = -r1 * y1 * y2 + alpha_param * r1 * xd1 * xd2
        dy3 = r1 * y1 * y2 - r2 * y3
        denom = xd1 * xd2 + xd3
        dy4 = 1.0 + ((3.0 * delta - y1 * y2 - y3) / denom) * np.exp(delta * y4)
        return [dy1, dy2, dy3, dy4]
    def phi(t):
        return [5.0, 0.1, 0.0, 0.0]
    def alpha(t, y):
        y1, y2, y3, y4 = y
        return t - y4
    t_span = [0.0, 40.0]
    solution = solve_dde(t_span, f, alpha, phi, method = method, Atol=Tol, Rtol=Tol)
    real_solution = None
    return solution, real_solution

def E1(method, Tol):
    r, c = np.pi/np.sqrt(3) + 1/20, np.sqrt(3)/(2*np.pi) - 1/25
    def f(t, y, x, z):
        return r*y*(1 - x - c*z)
    def phi(t):
        return 2 + t
    def phi_t(t):
        return 1
    def alpha(t, y):
        return t - 1
    beta = alpha
    t_span = [0, 40]
    solution = solve_ndde(t_span, f, alpha, beta, phi, phi_t, method = method, Atol=Tol, Rtol=Tol)
    real_solution = None
    return solution, real_solution

def E2(method, Tol):
    alpha_ = 1 / 10
    rho_   = 29 / 10
    tau_   = 21 / 50
    def f(t, y, x, z):
        y1, y2 = y
        x1, x2 = x
        z1, z2 = z
        dy1 = y1 * (1 - x1 - rho_ * z1) - (y2 * y1**2) / (y1**2 + 1)
        dy2 = y2 * ( (y1**2) / (y1**2 + 1) - alpha_ )
        return np.array([dy1, dy2])
    def phi(t):
        phi1 = 33 / 100 - t / 10
        phi2 = 111 / 50 + t / 10
        return np.array([phi1, phi2])
    def phi_t(t):
        dphi1 = -1 / 10
        dphi2 =  1 / 10
        return np.array([dphi1, dphi2])
    def alpha(t, y):
        return t - tau_
    beta = alpha  
    t_span = [0, 2]
    solution = solve_ndde(t_span, f, alpha, beta, phi, phi_t, method = method, Atol=Tol, Rtol=Tol)
    real_solution = None
    return solution, real_solution




def F1(method, Tol):
    def f(t, y, x, z):
        return 2*np.cos(2*t)*(x**(2*np.cos(t))) + np.log(z) - np.log(2*np.cos(t)) - np.sin(t)
    def phi(t):
        return 1
    def phi_t(t):
        return 2
    def alpha(t, y):
        return t/2
    beta = alpha
    t_span = [0, 1]
    solution = solve_ndde(t_span, f, alpha, beta, phi, phi_t, method = method, Atol=Tol, Rtol=Tol)
    def real_solution(t):
        return np.exp(np.sin(2*t))
    return solution, real_solution


def F2(method, Tol):
    def f(t, y, x, z):
        return z
    def phi(t):
        return np.exp(-t**2)
    def phi_t(t):
        return -2*t*np.exp(-t**2)
    def alpha(t, y):
        return 2*t - 0.5
    beta = alpha
    t_span = [0.25, 0.499]
    solution = solve_ndde(t_span, f, alpha, beta, phi, phi_t, method = method, Atol=Tol, Rtol=Tol)

    x = [(1 - 2**(-i))/2 for i in range(10)]
    B = [0]
    for i in range(1, 10):
        B.append(2*(4**(i-1) + B[i-1]))
    C = [0]
    for i in range(1, 10):
        C.append(-4**(i-2) - B[i-1]/2 + C[i-1])
    def y_piece(t, i=0):
        return np.exp(-4**i * t**2 + B[i]*t + C[i]) / (2**i) + K[i]
    K = [0]
    for j in range(1, 10):
        K.append(-np.exp(-4**j * x[j]**2 + B[j]*x[j] + C[j])/(2**j) + y_piece(x[j], i=j-1))
    def real_solution(t):
        for j in range(len(x)-1):
            if x[j] <= t <= x[j+1]:
                return y_piece(t, i=j)
        return y_piece(t, i=len(x)-2)  

    return solution, real_solution

def F3(method, Tol, L3 = 0.2):
    def alpha(t, y):
        return 0.5 * t * (1 - np.cos(2 * np.pi * t))
    beta = alpha  
    def f(t, y, x, z):
        return np.exp(-y) + L3 * (np.sin(z) - np.sin(1 / (3 + alpha(t, y))))
    def phi(t):
        return np.log(3.0)
    def phi_t(t):
        return 1/3.0
    t_span = [0.0, 10.0]
    solution = solve_ndde(t_span, f, alpha, beta, phi, phi_t, method = method, Atol=Tol, Rtol=Tol)
    def real_solution(t):
        return np.log(t + 3.0)
    return solution, real_solution

def F4(method, Tol):
    return F3(method, Tol, L3 = 0.4)


def F5(method, Tol):
    return F3(method, Tol, L3 = 0.6)


def G1(method, Tol):
    def f(t, y, x, z):
        return -z
    def phi(t):
        return 1 - t
    def phi_t(t):
        return -1
    def alpha(t, y):
        return t - (1/4)*y**2
    beta = alpha
    t_span = [0, 1]
    solution = solve_ndde(t_span, f, alpha, beta, phi, phi_t, method = method, Atol=Tol, Rtol=Tol)
    real_solution = lambda t: t + 1
    return solution, real_solution


def G2(method, Tol):
    def f(t, y, x, z):
        return -z
    def phi(t):
        return 1 - t
    def phi_t(t):
        return -1
    def alpha(t, y):
        return y - 2
    beta = alpha
    t_span = [0, 1]
    solution = solve_ndde(t_span, f, alpha, beta, phi, phi_t, method = method, Atol=Tol, Rtol=Tol)
    real_solution = lambda t: t + 1
    return solution, real_solution

def H1(method, Tol):
    def f(t, y, x, z):
        return -(4 * t * y**2) / (4 + (np.log(np.cos(2*t)))**2) + np.tan(2*t) + 0.5 * np.arctan(z)
    def phi(t):
        return 0
    def phi_t(t):
        return 0
    def alpha(t, y):
        return t*y**2 / (1 + y**2)
    beta = alpha
    t_span = [0, 0.225*np.pi]
    solution = solve_ndde(t_span, f, alpha, beta, phi, phi_t, method = method, Atol=Tol, Rtol=Tol)
    real_solution = lambda t: -np.log(np.cos(2*t))/2
    return solution, real_solution



def H2(method, Tol, L3 = 0.1):
    L3 = 0.1
    def f(t, y, x, z):
        val = np.cos(t)*(1 + x)+L3*y*z + (1 - L3)*np.sin(t)*np.cos(t*np.sin(t)**2) - np.sin(t + t*np.sin(t)**2)
        return val
    def phi(t):
        return 0
    def phi_t(t):
        return 1
    def alpha(t, y):
        return t*(y**2)
    beta = alpha
    t_span = [0, np.pi]
    solution = solve_ndde(t_span, f, alpha, beta, phi, phi_t, method = method, Atol=Tol, Rtol=Tol)
    real_solution = lambda t: np.sin(t)
    return solution, real_solution


def H3(method, Tol):
    return H2(method, Tol, L3 = 0.3)


def H4(method, Tol):
    return H2(method, Tol, L3 = 0.5)


problems = {'A1':A1, 'A2':A2, 'B1':B1, 'B2':B2, 'C1':C1, 'C2':C2, 'C3':C3, 'C4':C4, 'D1':D1, 'D2':D2, 'E1':E1, 'E2':E2, 'F1':F1, 'F2':F2, 'F3':F3, 'F4':F4, 'F5':F5, 'G1':G1, 'G2':G2, 'H1':H1, 'H2':H2, 'H3':H3, 'H4':H4}
# problems = {'B1':B1, 'B2':B2, 'D1':D1, 'F1':F1, 'F2':F2, 'F3':F3, 'F4':F4, 'F5':F5, 'G1':G1, 'G2':G2, 'H1':H1, 'H2':H2, 'H3':H3, 'H4':H4}
# problems = {'B2': B2, 'G1':G1}

methods = ['CERK3', 'CERK4', 'CERK5']
tolerances = [1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12]


for Tol in tolerances:
    print('======================== Overall Statistcs for all DDETST ===========================')
    print(f'Tol = {Tol} \n')
    for method in methods:

        total_steps = 0
        total_fails = 0
        total_feval = 0
        print(f'method = {method}\n')
        for problem_name, problem in problems.items():
            # print(f' Problem: {problem_name}')
            solution, real_solution = problem(method, Tol)
            total_steps += solution.steps
            total_fails += solution.fails
            total_feval += solution.feval
            if solution.status == 'failed':
                print('iter', iter)
                print('problem', problem)
                input('failed')
            # if real_solution is None:
            #     print('No analitical solution')
            # else:
            #     real_solution_at_grid = np.array([real_solution(t) for t in solution.t])
            #     approx_solution_at_grid = np.array([solution.eta(t) for t in solution.t])
                # print('----')
                # print('real_sol', real_solution_at_grid)
                # print('----')
                # print('aprox', approx_solution_at_grid)
                # print('----')
                # grid_error = np.max(np.abs(real_solution_at_grid - approx_solution_at_grid))
                # print('grid error: ', grid_error)
        print('total steps: ', total_steps)
        print('total fails: ', total_fails)
        print('total feval: ', total_feval)
        print('')
