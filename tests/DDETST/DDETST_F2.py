import matplotlib.pyplot as plt
import numpy as np

from DDE_solver.rkh_refactor import *


def f(t, y, x, z):
    return z

def phi(t):
    return np.exp(-t**2)

def phi_t(t):
    return -2*t*np.exp(-t**2)

def alpha(t, y):
    return 2*t - 0.5

beta = alpha

# ---- Analytical construction ----

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

def real_sol(t):
    for j in range(len(x)-1):
        if x[j] <= t <= x[j+1]:
            return y_piece(t, i=j)
    return y_piece(t, i=len(x)-2)  # fallback for boundary

t_span = [0.25, 0.499]







print(f'{'='*80}')
print(f''' {'='*80} 
      This is problem 1.3.4 from Paul
      ''')

methods = ['RKC3', 'RKC4', 'RKC5']
tolerances = [1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12]
# methods = ['RKC4', 'RKC5']
# tolerances = [1e-2,  1e-4, 1e-6, 1e-8, 1e-10]

for Tol in tolerances:
    print('===========================================================')
    print(f'Tol = {Tol} \n')
    for method in methods:
        solution = solve_ndde(t_span, f, alpha, beta, phi, phi_t, method = method, Atol=Tol, Rtol=Tol)

        max_diff = 0
        for i in range(len(solution.t) - 1):
            tt = np.linspace(solution.t[i], solution.t[i + 1], 100, endpoint=True)
            sol = np.array([solution.eta(i) for i in tt])
            realsol = np.array([real_sol(i) for i in tt])
            max_diff = np.max(np.abs(realsol - sol))
            if max_diff > max_diff:
                max_diff = max_diff

            # if len(solution.discs) > 1:
            #     print('tt', tt)
            #     print('discs', solution.discs)
            #     print('t[i] = ', solution.t[i])
            #     print('t[i+1] = ', solution.t[i+1])
            #     input(solution.discs[1] in tt)

        exact_discs = x[1:len(solution.discs) + 1]
        disc_diff = np.array(solution.discs) - np.array(exact_discs)
        disc_diff = np.max(np.abs(disc_diff))

        print(f'method = {method}')
        print('max diff', max_diff)
        print('steps: ', solution.steps)
        print('fails: ', solution.fails)
        print('feval: ', solution.feval)
        print('no. discs found: ', len(solution.discs))
        print('diff discs found: ', disc_diff)
        print('')

        # t = np.array(solution.t)      
        # h = np.diff(t)
        #
        # plt.plot(t[1:], h, marker='o')
        # plt.scatter(exact_discs[1:], np.zeros_like(exact_discs[1:]))
        # plt.xlabel("t_n")
        # plt.ylabel("Step size h_n = t_n - t_{n-1}")
        # plt.title("Adaptive step sizes")
        # plt.show()
