import numpy as np
from scipy.optimize import root


def get_discs(alpha, tspan, h0=0.01, max_it=1000):
    t_0 = tspan[0]
    discs = []
    h = h0
    it = 0

    while it < max_it and t_0 <= tspan[-1]:
        # Define the function alpha(t) - t_0
        def f(t):
            return alpha(t) - t_0

        # Root-finding with initial guess h
        sol = root(f, x0=t_0 + h)
        if not sol.success:
            break  # stop if root-finding failed

        t_1 = sol.x[0]
        discs.append(t_1)

        # Update for next iteration
        h = t_1 - t_0
        t_0 = t_1
        it += 1

    return discs


alpha = lambda t: np.cos(t)
tspan = [1.0, 10.0]
discs = get_discs(alpha, tspan)
print("Example 3 (cosine):", discs)


def DDEsolve(ddefun, delays, history, tspan):
    # returns a function that is the solution
    solution = []
    return solution
