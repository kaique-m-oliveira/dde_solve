import matplotlib.pyplot as plot
import numpy as np
import scipy as sc


def alpha(t):
    return t - 1


def phi(t):
    return 1


def f(t, y, alpha):
    return y(alpha(t)) + alpha(t + 1)


def get_interval(t_0, alpha):  # pretested
    root = sc.optimize.fsolve(lambda t: alpha(t) - t_0, t_0)
    return root[0]  # the root is a list of one element


def get_discontinuities(t_0, t_f, alpha):  # pretested
    disc = [t_0]
    while disc[-1] < t_f:
        disc.append(get_interval(disc[-1], alpha))
    return disc  # tracking discontinuities


def interpolate(t, y):
    return sc.interpolate.interp1d(t, y)  # WARN: returns a function


def DDE_solve(t, f, phi, alpha, h):
    # this line should be enough to get all intervals for iteration
    disc = get_discontinuities(t[0], t[-1], alpha)
    interpol = [phi]
    tt = np.arange(t[0] - (t[-1] - t[0]) / 8, t[0], h)
    # yy = [phi(t) for t in tt]
    yy = np.zeros(len(tt))

    for n in range(len(disc) - 1):
        t = np.arange(disc[n], disc[n + 1], h)
        fun = lambda x, y: f(x, interpol[-1], alpha)
        # sol = sc.integrate.solve_ivp(fun, [disc[n], disc[n + 1]], [disc[n]], t_eval=t)
        sol = sc.integrate.odeint(fun, disc[n], t)
        tt = np.append(tt, t)
        yy = np.append(yy, sol)
        plot.plot(tt, yy)
        plot.show()
        # tt = np.append(tt, sol.t)
        # yy = np.append(yy, sol.y)
        # interpol.append(interpolate(sol.t, sol.y))

    return interpol, tt, yy


t_0, t_f, h = 0, 10, 10 ** (-3)


inter, tt, yy = DDE_solve([t_0, t_f], f, phi, alpha, h)

print("len(tt)", len(tt), "tt", tt)
print("len(yy)", len(yy), "yy", yy)
plot.plot(tt, yy)
plot.show()
