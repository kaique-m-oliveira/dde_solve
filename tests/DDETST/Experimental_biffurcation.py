""" This is an experimental feature, still needs more development"""

import numpy as np

from DDE_solver.rkh_refactor import *


def f(t, y, yq):
    return yq + 1/2


def phi(t):
    return 1 if t <= -1 else 0


def alpha(t, y):
    return t - abs(y) - 1


def real_sol_1(t):
    return 3*t/2


def real_sol_2(t):
    return t/2


t_span = [0, 2]
discs = [(-1, 1, 0)]


methods = ['RKC3', 'RKC4', 'RKC5']
tolerances = [1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12]


for Tol in tolerances:
    print('===========================================================')
    print(f'Tol = {Tol} \n')
    for method in methods:


        tt = np.linspace(t_span[0], t_span[1], 100)
        realsol1 = np.array([real_sol_1(t) for t in tt])
        realsol2 = np.array([real_sol_2(t) for t in tt])

        solutionList = solve_dde(t_span, f, alpha, phi, discs=discs, method=method, Atol=Tol, Rtol=Tol)

        max_diff1 = 0
        max_diff2 = 0
        for i in range(len(solution.t) - 1):
            tt = np.linspace(solution.t[i], solution.t[i + 1], 100)
            sol1 = [solutionList.solutions[0].eta(i) for i in tt]
            sol2 = [solutionList.solutions[1].eta(i) for i in tt]
            realsol = np.array([real_sol(i) for i in tt])
            max_diff_11 = np.max(np.abs(realsol - sol))
            max_diff_22 = np.max(np.abs(realsol - sol))
            if max_diff > max_diff:
                max_diff1 = max_diff
                max_diff2 = max_diff
        
        print(f'method = {method}')
        print('max diff', max_diff)
        print('steps: ', solution.steps)
        print('fails: ', solution.fails)
        print('feval: ', solution.feval)
        print('')

