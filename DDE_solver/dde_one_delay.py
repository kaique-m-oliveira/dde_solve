import ast
import inspect
import random

import matplotlib.pyplot as plt
import numpy as np
import scipy as sc
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicHermiteSpline, CubicSpline
from scipy.optimize import fsolve


def find_delay(f):
    """returns a list with the delays"""
    source_code = inspect.getsource(f)
    tree = ast.parse(source_code)

    delay_funcs, delay_strings = [], []

    for node in ast.walk(tree):
        if isinstance(node, ast.Return):
            # Find all Call nodes within the return statement
            call_nodes = [n for n in ast.walk(node.value) if isinstance(n, ast.Call)]
            for call_node in call_nodes:
                arg_node = call_node.args[0]
                if isinstance(arg_node, ast.BinOp):
                    delay_string = ast.unparse(arg_node)
                    delay_funcs.append(lambda t, f=delay_string: eval(f))
                    delay_strings.append(delay_string)
    print("the delays are:", delay_strings)
    return delay_funcs


def get_primary_discontinuities(t_span, delays):
    """returns the primary discontinuities in the interval t from the delays"""
    t0, tf = t_span
    N = len(delays)
    discontinuities = [[t0] for i in range(N)]
    for i in range(N):
        h = 10
        t = t0
        f = lambda x: delays[i](x) - t
        while t < tf and h > 10**-5:
            disc = fsolve(f, t + 0.01)[0]
            if disc < t:
                print("disc < t, deu ruim")
                return

            # FIX: this part messes up the code
            # if disc <= tf:
            #     discontinuities[i].append(disc)
            discontinuities[i].append(disc)
            h = disc - t
            t = disc
            # print(disc)
            # print("t", t, "tf", tf, "t < tf", t < tf)
            # print(f"h = {h} is less then 10**-5 {h < 10**-5}")
    return discontinuities


def get_yq(discs, x, sol):

    for i in range(len(discs)):
        if x <= discs[i]:
            return sol[i](x)
    if x <= tf:
        return sol[-1]
    print(f"{x} is not in the interval t")


def get_yqq(discs, x, sol):

    if len(discs) == 1:
        return sol[0]
    for i in range(len(discs)):
        if x <= discs[i]:
            return sol[i]
    if x <= tf:
        return sol[-1]
    print(f"{x} is not in the interval t")


def f1(t, y, yq):
    return -yq


def f2(t, y):
    return -np.sin(t - np.pi / 2)


def f3(t, y, yq):
    return -y + yq



def delay1(t):
    return t - 1


def delay2(t):
    return t - np.pi / 2


def phi1(t):
    return 1


def phi2(t):
    return np.sin(t)


def rk4_arit(f, t0, t1, y0):
    h = t1 - t0
    k1 = h * f(t0, y0)
    k2 = h * f(t0 + h / 2, y0 + k1 / 2)
    k3 = h * f(t0 + h / 2, y0 + k2 / 2)
    k4 = h * f(t0 + h, y0 + k3)
    y1 = y0 + (k1 + 2 * k2 + 2 * k3 + k4) / 6
    # print("-" * 40, "disc at t = ", t1, "-" * 40)
    # print("k1 = ", k1)
    # print("k2 = ", k2)
    # print("k3 = ", k3)
    # print("k4 = ", k4)
    # print("y1 = ", y1)
    return y1


def rk4_arit_delay(f, t0, t1, y0, yq, delay):
    h = t1 - t0
    k1 = h * f(t0, y0, yq(delay(t0)))
    k2 = h * f(t0 + h / 2, y0 + k1 / 2, yq(delay(t0 + h / 2)))
    k3 = h * f(t0 + h / 2, y0 + k2 / 2, yq(delay(t0 + h / 2)))
    k4 = h * f(t0 + h, y0 + k3, yq(delay(t0 + h)))
    y1 = y0 + (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return y1


def rk4(f, t_span, y0, t):
    """as the name suggests"""

    N = len(t)
    y = np.zeros(N)
    y[0] = y0
    for n in range(N - 1):
        y[n + 1] = rk4_arit(f, t[n], t[n + 1], y[n])
    return y


def rk4_while(f, t_span, y0, h):
    """as the name suggests"""

    t0, tf = t_span
    t, y = [t0], [y0]

    while t[-1] < tf:
        y.append(rk4_arit(f, t[-1], t[-1] + h, y[-1]))
        t.append(t[-1] + h)
    sol = CubicSpline(t, y)
    return sol


def rk4_while_yq(f, t_span, y0, h):
    """as the name suggests"""

    t0, tf = t_span
    t, y = [t0], [y0]

    while t[-1] < tf:
        yq = np.sin(t[-1] - np.pi / 2)
        y.append(rk4_arit_delay(f, t[-1], t[-1] + h, y[-1], yq))
        t.append(t[-1] + h)
    sol = CubicSpline(t, y)
    return sol


def rk4_cont(f, t_span, phi, y0, delay, h):
    """for testing the rk_cont"""
    discs = get_primary_discontinuities(t_span, [delay])[0]
    print("list of discs", discs)
    print("t_span", t_span)
    t0 = t_span[0]
    sol = [phi]
    y = [phi(t0)]
    t = [t0]
    for disc in discs:
        y = [y[-1]]
        t = [t[-1]]
        while delay(t[-1]) < disc:  # adding to the discrete solution

            x = delay(t[-1])
            yq = get_yqq(discs, x, sol)

            # WARN:  y é diferente do  yy por causa dos k2, k3, k4
            y.append(rk4_arit_delay(f, t[-1], t[-1] + h, y[-1], yq, delay))

            t.append(t[-1] + h)
        sol.append(CubicSpline(t, y))

    def solution(var):
        if len(discs) == 1:
            return sol[0](var) if var <= discs[0] else sol[1](var)

        for i in range(len(discs)):
            if var <= discs[i]:
                return sol[i](var)
        print(f"1 the point {var} isn't part of the domain of the solution")

    return solution


def rk4_cont_test(f, ff, t_span, phi, y0, delay, h):
    """for testing the rk_cont"""
    discs = get_primary_discontinuities(t_span, [delay])[0]
    print("list of discs", discs)
    print("t_span", t_span)
    t0 = t_span[0]
    sol = [phi]
    ssol = [phi]
    y = [phi(t0)]
    yy = [y0]
    t = [t0]
    for disc in discs:
        y = [y[-1]]
        yy = [yy[-1]]
        t = [t[-1]]
        while delay(t[-1]) < disc:  # adding to the discrete solution

            x = delay(t[-1])
            yq = get_yqq(discs, x, sol)

            # WARN:  y é diferente do  yy por causa dos k2, k3, k4
            y.append(rk4_arit_delay(f, t[-1], t[-1] + h, y[-1], yq, delay))
            yy.append(rk4_arit(ff, t[-1], t[-1] + h, yy[-1]))

            t.append(t[-1] + h)
        sol.append(CubicSpline(t, y))
        ssol.append(CubicSpline(t, yy))

    def solution1(var):
        if len(discs) == 1:
            return sol[0](var) if var <= discs[0] else sol[1](var)

        for i in range(len(discs)):
            if var <= discs[i]:
                return sol[i](var)
        print(f"1 the point {var} isn't part of the domain of the solution")

    def solution2(var):
        if len(discs) == 1:
            return ssol[0](var) if var <= discs[0] else ssol[1](var)

        for i in range(len(discs)):
            if var <= discs[i]:
                return ssol[i](var)
        print(f"2 the point {var} isn't part of the domain of the solution")

    return solution1, solution2


def rk4_cont_before_tests(f, t_span, phi, delay, h):
    """who tf knows"""
    discs = get_primary_discontinuities(t_span, [delay])[0]
    sol = [phi]

    t0 = t_span[0]
    y = [phi(t0)]
    t = [t0]
    # dy = [0]
    # print("the discs are", discs)
    for disc in discs:
        y = [y[-1]]
        t = [t[-1]]
        # dy = [dy[-1]]
        # print("disc = ", disc)
        while delay(t[-1]) < disc:  # adding to the discrete solution
            # print("inside")
            x = delay(t[-1])
            # print(f"x is {x} and t is {t[-1]}")
            yq = get_yq(discs, x, sol)
            # print("-" * 99)
            # print("yq =", yq, "np.sin(t - np.pi/2) = ", np.sin(t[-1] - np.pi / 2))
            # print("-" * 99)
            y.append(rk4_arit_delay(f, t[-1], t[-1] + h, y[-1], yq))
            # dev = (y[-1] - y[-2]) / h
            # dy.append(dev)
            t.append(t[-1] + h)
        # print("t =", t)
        # print("y =", y)
        sol.append(CubicSpline(t, y))
        # print("sol = ", sol)
        # sol.append(CubicHermiteSpline(t, y, dy))

    def solution(var):
        if len(discs) == 1:
            return sol[0](var) if var <= discs[0] else sol[1](var)

        for i in range(len(discs)):
            if var <= discs[i]:
                return sol[i](var)
        print(f"the point {var} isn't part of the domain of the solution")

    return solution


# NOTE: Continuous case
t0, tf = 0, 3
t_span = [t0, tf]
# t = np.arange(0, 100, 0.1)
h = 0.01
t = np.arange(t0, tf, h)


# TEST: second test (the combined version)
# WARN: O problema é no sol2
y0 = 0
sol1 = rk4_cont(f3, t_span, phi2, y0, delay1, h)
y = [sol1(i) for i in t]


# print("len of t", len(t))


print("len(t), len(y)", len(t), len(y))
plt.plot(t, y, label="cont", color="green")
plt.legend()
plt.show()


# TEST: this is the first test
"""
sol = rk4_cont(f, t_span, phi, delay, h)
y0 = 0
sol_simple = rk4_while(f_simple, t_span, y0, h)
sol_simple_yq = rk4_while_yq(f, t_span, y0, h)

y = [sol(i) for i in t]
y_simple = [sol_simple(i) for i in t]
y_simple_yq = [sol_simple_yq(i) for i in t]
sin = np.sin(t)
# WARN: this error analysis suggests the method is linear
error = 0
error2 = 0
error3 = 0
for i in range(100):
    x = random.uniform(t0, tf)
    diff = abs(np.sin(x) - sol(x))
    diff2 = abs(np.sin(x) - sol_simple(x))
    diff3 = abs(np.sin(x) - sol_simple_yq(x))
    # print(diff)
    if diff > error:
        error = diff
    if diff2 > error2:
        error2 = diff2
    if diff3 > error3:
        error3 = diff3

# print("len of t", len(t))
print("max error cont", error)
print("max error disc", error2)
print("max error disc yq", error3)


plt.plot(t, y, label="cont", color="green")
plt.plot(t, sin, label="sin", color="red")
plt.plot(t, y_simple, label="simple", color="blue")
plt.plot(t, y_simple_yq, label="simple yq", color="yellow")
plt.legend()
plt.show()
"""


# NOTE: discrete case


# # WARN:  THIS IS NOT GONNA WORK WITH SOLVE_IVP UNLESS I FIGURE OUT HOW TO USE THE PHI INSIDE F
# def DDE_solve(fun, t_span, phi, t_eval):
#     delays = find_delay(fun)
#     discs = get_primary_discontinuities(t_span, delays)[0]
#     sol = [phi]
#     N = len(discs)
#     for i in range(N):
#         sol = solve_ivp(fun, t_span, sol[i](discs[i]), method="RK45", t_eval=t_eval)
#     return sol


# def get_disc_euler(t_span, delay):
#     t0, tf = t_span
#     h = 10
#     t = t0
#     f = lambda x: delay(x) - t
#     discs = [t0]
#     while t < tf and h > 10**-5:
#         disc = fsolve(f, t + 0.01)[0]
#         if disc < t:
#             print("disc < t, deu ruim")
#             return
#         discs.append(disc)
#         h = disc - t
#         t = disc
#     return discs


def g(t, y):
    return 3 * y


# if __name__ == "__main__":
# phi = lambda t: 1
# t = np.arange(0, 100, 0.01)
# delay = find_delay(f)
# print(delay)
# discs = get_primary_discontinuities([0, 100], delay)[0]
# print(discs)
