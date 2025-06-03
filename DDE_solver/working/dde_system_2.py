import random

import matplotlib.pyplot as plt
import numpy as np
import scipy as sc
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicHermiteSpline, CubicSpline
from scipy.optimize import fsolve


def SIR(t, SI, SIq):
    A = 0.94
    a1, a2 = 0.5, 0.5
    u, b = 0.05, 0.1
    a, y = 0.5, 0.5
    tau = 1

    S, I = SI
    Sq, Iq = SIq
    DI = A - u * S - (b * S * I) / (1 + a1 * S + a2 * I)
    DS = (b * np.exp(-u * tau) * Sq * Iq) / (1 + a1 * Sq + a2 * Iq) - (u + a + y) * I
    return np.array([DI, DS])


def ex142_paul(t, xyz, xyzq):
    x, y, z = xyz
    xq, yq, zq = xyzq
    Dx = -x * yq + y
    Dy = x * yq - yq
    Dz = y - yq
    return np.array([Dx, Dy, Dz])


def exMATLAB(t, xyz, xyzq):
    x, y, z = xyz
    xq, yq, zq = xyzq
    Dx = xq
    Dy = xq + yq
    Dz = y
    return np.array([Dx, Dy, Dz])


def phi(t):
    return np.array([1, 1, 1])


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

            discontinuities[i].append(disc)
            h = disc - t
            t = disc
            # print(disc)
            # print("t", t, "tf", tf, "t < tf", t < tf)
            # print(f"h = {h} is less then 10**-5 {h < 10**-5}")
    return discontinuities


def get_yq_sys(discs, x, sol):
    for i in range(len(discs)):
        if x <= discs[i]:
            return np.array([sol[i][j](x) for j in range(len(sol[-1]))])
    print(f"{x} is not in the interval t")


def get_yq(discs, x, sol):

    for i in range(len(discs)):
        if x <= discs[i]:
            return sol[i](x)
    print(f"{x} is not in the interval t")


# def f(t, y, yq):
#     return -yq


def delay(t):
    return t - 1


def rk4_arit_delay(f, t0, t1, y0, yq):
    # print("rk4_arit_delay: ", "yq", yq, "f(t0, y0, yq)", f(t0, y0, yq))
    h = t1 - t0
    k1 = h * f(t0, y0, yq)
    k2 = h * f(t0 + h / 2, y0 + k1 / 2, yq + k1 / 2)
    k3 = h * f(t0 + h / 2, y0 + k2 / 2, yq + k2 / 2)
    k4 = h * f(t0 + h, y0 + k3, yq + k3)
    y1 = y0 + (k1 + 2 * k2 + 2 * k3 + k4) / 6
    # print(f"y = {y1} , t0 {t0}, t1 {t1}, y0 {y0} , yq {yq}")
    return y1


def rk4_cont_sys(f, t_span, phi, delay, h):
    """who tf knows"""
    discs = get_primary_discontinuities(t_span, [delay])[0]
    sol = [phi]

    t0 = t_span[0]
    y = np.zeros((1, len(phi(t0))))
    y[0] = phi(t0)
    t = [t0]
    # dy = [0]
    # print(f"y = {len(y)} dy = {len(dy)}")
    for disc in discs:
        y = [y[-1]]
        t = [t[-1]]
        # dy = [dy[-1]]
        # i = 0
        while delay(t[-1]) < disc:  # adding to the discrete solution
            x = delay(t[-1])
            # print(f"x is {x} and t is {t[-1]}")
            yq = get_yq(discs, x, sol)
            rk4_stuff = rk4_arit_delay(f, t[-1], t[-1] + h, y[-1], yq)
            # print("yq", yq, "rk4_stuff", rk4_stuff, "y", y)
            y = np.vstack((y, rk4_stuff))
            # dev = (y[-1] - y[-2]) / h
            # print(f"t = {t[-1]} and dy = {dev}")
            # dy.append(dev)
            t.append(t[-1] + h)
            # print(f" dy at t = {t[-1]} is {(sol[-1](t0 + h) - sol[-1](t0)) / h}")
            # print(f" len(y) {len(y)} len(dy) {len(dy)}")
        # print(f"y = {len(y)} dy = {len(dy)}")
        # sol.append(CubicSpline(t, y))
        s = []
        for i in range(len(y[-1])):
            # NOTE: works
            s.append(CubicSpline(t, y[:, i]))
        #     print("last y", y[:, i][-1])
        # print("bla", s[0](1.5), s[1](1.5))
        sol.append(lambda t: np.array([s[i](t) for i in range(len(s))]))

        # print("sol", sol)
        # print("sol[0](0.5)", sol[0](0.5), "sol[1](1.5)", sol[1](0.5))
        # print(f"começo e final de t, {t[0]} e {t[-1]}")

    # WARN: here
    #
    # print("-" * 99)
    # print("to the solution")
    # print("-" * 99)
    # print("what is sol", len(sol))

    def solution(var):

        for i in range(len(discs)):
            if var <= discs[i]:
                return sol[i](var)

    return solution


def rk4_cont(f, t_span, phi, delay, h):
    """who tf knows"""
    discs = get_primary_discontinuities(t_span, [delay])[0]
    sol = [phi]

    t0 = t_span[0]
    y = [phi(t0)]
    t = [t0]
    dy = [0]
    # print(f"y = {len(y)} dy = {len(dy)}")
    for disc in discs:
        y = [y[-1]]
        t = [t[-1]]
        dy = [dy[-1]]
        while delay(t[-1]) < disc:  # adding to the discrete solution
            x = delay(t[-1])
            # print(f"x is {x} and t is {t[-1]}")
            yq = get_yq(discs, x, sol)
            y.append(rk4_arit_delay(f, t[-1], t[-1] + h, y[-1], yq))
            dev = (y[-1] - y[-2]) / h
            # print(f"t = {t[-1]} and dy = {dev}")
            dy.append(dev)
            t.append(t[-1] + h)
            # print(f" dy at t = {t[-1]} is {(sol[-1](t0 + h) - sol[-1](t0)) / h}")
            # print(f" len(y) {len(y)} len(dy) {len(dy)}")
        # print(f"y = {len(y)} dy = {len(dy)}")
        # sol.append(CubicSpline(t, y))
        sol.append(CubicHermiteSpline(t, y, dy))
        # print(f"começo e final de t, {t[0]} e {t[-1]}")

    def solution(var):
        for i in range(len(discs)):
            if var <= discs[i]:
                return sol[i](var)

    return solution


t0, tf = 0, 10
t_span = [t0, tf]
h = 0.01
t = np.arange(t0, tf, h)
fuck = rk4_cont_sys(exMATLAB, t_span, phi, delay, h)

S, I, R = np.zeros(len(t)), np.zeros(len(t)), np.zeros(len(t))
for i in range(len(t)):
    S[i], I[i], R[i] = fuck(t[i])

plt.plot(t, S, label="S", color="red")
plt.plot(t, I, label="I", color="blue")
plt.plot(t, R, label="R", color="green")
plt.show()
# print("I dont think this is gonnna work at all", sol)
