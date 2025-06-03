import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import fsolve


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
                print("discontinuity < t")
                return

            discontinuities[i].append(disc)
            h = disc - t
            t = disc
    return discontinuities


def get_yq(discs, x, sol):

    if len(discs) == 1:
        return sol[0]
    for i in range(len(discs)):
        if x <= discs[i]:
            return sol[i]
    if x <= tf:
        return sol[-1]
    print(f"{x} is not in the interval t")


def rk4_arit_delay(f, t0, t1, y0, yq, delay):
    h = t1 - t0
    k1 = h * f(t0, y0, yq(delay(t0)))
    k2 = h * f(t0 + h / 2, y0 + k1 / 2, yq(delay(t0 + h / 2)))
    k3 = h * f(t0 + h / 2, y0 + k2 / 2, yq(delay(t0 + h / 2)))
    k4 = h * f(t0 + h, y0 + k3, yq(delay(t0 + h)))
    y1 = y0 + (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return y1


def rk4_delay(f, phi, delay, t_span, h):
    """continuous rk4 with cubic spline for time dependent delayed differential equations"""
    discs = get_primary_discontinuities(t_span, [delay])[0]
    t0 = t_span[0]
    sol = [phi]
    y = [phi(t0)]
    t = [t0]
    for disc in discs:
        y = [y[-1]]
        t = [t[-1]]
        while delay(t[-1]) < disc:  # adding to the discrete solution
            x = delay(t[-1])
            yq = get_yq(discs, x, sol)
            y.append(rk4_arit_delay(f, t[-1], t[-1] + h, y[-1], yq, delay))
            t.append(t[-1] + h)
        sol.append(CubicSpline(t, y))

    def solution(var):
        if len(discs) == 1:
            return sol[0](var) if var <= discs[0] else sol[1](var)

        for i in range(len(discs)):
            if var <= discs[i]:
                return sol[i](var)
        print(f"the point {var} isn't part of the domain of the solution")

    return solution


if __name__ == "__main__":
    t0, tf = 0, 3
    t_span = [t0, tf]
    # t = np.arange(0, 100, 0.1)
    h = 0.01
    t = np.arange(t0, tf, h)

    f1 = lambda t, y, yq: -yq
    delay1 = lambda t: t - 1
    phi1 = lambda t: 1

    f2 = lambda t, y: -np.sin(t - np.pi / 2)
    delay2 = lambda t: t - np.pi / 2
    phi2 = lambda t: np.sin(t)

    f4 = lambda t, y, yq: 3 * yq * (1 + y)
    phi4 = lambda t: t
    delay4 = lambda t: t - 1

    f5 = lambda t, y, yq: y + yq + 3 * np.cos(t) + 5 * np.sin(t)
    phi5 = lambda t: 3 * np.sin(t) - 5 * np.cos(t)
    delay5 = lambda t: t - np.pi
    real_sol_5 = lambda t: 3 * np.sin(t) - 5 * np.cos(t)
    # y5 = [real_sol_5(i) for i in t]
    # plt.plot(t, y5, label="real solution", color="red")

    f6 = lambda t, y, yq: (1 - np.exp(-3 * np.pi / 2)) * y + yq - np.sin(t)
    phi6 = lambda t: np.exp(t) + np.sin(t)
    delay6 = lambda t: t - 3 * np.pi / 2
    real_sol_6 = lambda t: np.exp(t) + np.sin(t)
    # y6 = [real_sol_6(i) for i in t]
    # plt.plot(t, y6, label="real solution", color="red")

    f7 = lambda t, y, yq: yq
    phi7 = lambda t: 1
    delay7 = lambda t: t - 1

    f8 = lambda t, y, yq: (1 - np.exp(-3 * np.pi / 2)) * y + yq - np.sin(t)
    phi8 = lambda t: np.exp(t) + np.sin(t)
    delay6 = lambda t: t - 3 * np.pi / 2
    real_sol_6 = lambda t: np.exp(t) + np.sin(t)

    sol = rk4_delay(f7, phi7, delay7, t_span, h)
    y = [sol(i) for i in t]
    plt.plot(t, y, label="aprox. solution", color="green")
    plt.legend()
    plt.show()
