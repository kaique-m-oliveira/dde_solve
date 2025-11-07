import numpy as np

from DDE_solver.rkh_refactor import *


def f(t, y, x):
    return 1 - x

def phi(t):
    return np.log(t)

def alpha(t, y):
    return np.exp(1 - 1/t)

def real_sol(t):
    return np.log(t)

t_span = [0.1, 10]

print('alpha0', alpha(0.5, phi(0.5)))


print(f'{'='*80}')
print(f''' {'='*80} 
      This is problem 1.2.6 from Paul
      ''')
methods = ['RKC3', 'RKC4','RKC5']
tolerances = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-8, 1e-10, 1e-12]
# methods = ['RKC4', 'RKC5']
# tolerances = [1e-4, 1e-6, 1e-8, 1e-10, 1e-12]


for Tol in tolerances:
    print('===========================================================')
    print(f'Tol = {Tol} \n')
    for method in methods:
        solution = solve_dde(f, alpha, phi, t_span, method = method, Atol=Tol, Rtol=Tol)

        max_diff = 0
        for i in range(len(solution.t) - 1):
            tt = np.linspace(solution.t[i], solution.t[i + 1], 100)
            sol = np.array([solution.eta(i) for i in tt])
            realsol = np.array([real_sol(i) for i in tt])
            max_diff = np.max(np.abs(realsol - sol))
            if max_diff > max_diff:
                max_diff = max_diff
        
        print(f'method = {method}')
        print('max diff', max_diff)
        print('steps: ', solution.steps)
        print('fails: ', solution.fails)
        print('feval: ', solution.feval)
        print('')
        # input('finished')
        # t_plot = np.linspace(t_span[0], t_span[-1], 1000)
        # approx_plot =  [solution.eta(i) for i in t_plot]
        # realsol = [real_sol(t) for t in t_plot]
        # plt.plot(t_plot, approx_plot, color="blue", label='aproxx')
        # plt.plot(t_plot, realsol, color="red", label='real')
        # plt.legend()
        # plt.show()
        

# t_plot = np.linspace(t_span[0], t_span[-1], 1000)
# approx_plot =  [solution.eta(i) for i in t_plot]
# plt.plot(t_plot, approx_plot, color="blue", label='aproxx')
# plt.legend()
# plt.show()
