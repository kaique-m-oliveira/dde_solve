import ast
import inspect

import numpy as np
import scipy


def f(t, y):
    return y(t - 1)  # + y(t - 4)


def phi(t, y):
    return 1


def find_delay(f):
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
    return delay_funcs, delay_strings


def get_roots(t_0, f):
    "get's the f(t) = t_0"
    history = []
    for fun in f:
        f = lambda t: fun(t) - t_0
        roots = scipy.optimize.fsolve(f, t_0)
        print(roots)
        if len(roots) == 0:
            history.append(None)
            continue
        # roots.sort() #FIX: might be useless but let's keep it for now
        for x in roots:
            if x > t_0:
                history.append(x)
                break
        # history.append(None)  #FIX: might be useless but let's keep it for now
    return history


# WARN: all of this is cancer
def get_history(t, y, f, delay_funcs, history):
    phi_final = []
    N = len(t)
    for i in range(N - 1):
        phis = []
        for fun in delay_funcs:
            ff = lambda t: fun(t) - t[i + 1]
            roots = scipy.optimize.fsolve(ff, t[-1])
            if len(roots) == 0 or not np.isclose(fun(roots[0])):
                phis.append(None)
                continue
            roots.sort()
            for x in roots:
                if x > t[i + 1]:
                    phis.append(x)
                    break
            phis.append(None)
        phi_final.append(phis)
    return phi_final


# WARN: all of this is cancer
def dde_solver(init, f):
    t_00, t_0, t_f, h, phi = init  # init will take these
    n = (t_f - t_0) / h
    delay_funcs, delay_strings = find_delay(f)

    tt = np.linspace(t_0, t_f, n)
    while True:
        t = [t_00, t_0]
        y = [phi(t, None)]  # FIX: uma merda, deve ser arrumado
        while t < t_f:
            history = get_roots(t[-1], delay_funcs)
            ode = get_ode(t, y, f, delay_funcs, history)
            y.append(scipy.integrate.odeint(f, y[-1], tt))
        return y


def test_root(f):
    for t0 in range(0, 100):
        roots = get_roots(t0, f)
        print(roots)


if __name__ == "__main__":
    delay_funcs, delay_strings = find_delay(f)
    test_root([lambda t: (t - 3) * (t + 3)])
