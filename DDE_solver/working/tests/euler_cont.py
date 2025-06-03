import ast
import inspect

import numpy as np


# NOTE: returns a tuple of lists
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


def f(t, y):
    return y(t - 1)


def phi(t):
    return 1


def euler(t_vec, f, phi):
    N = len(t_vec)
    y = np.zeros(N)
    for i in range(N):
        y[i + 1] = y[i] + (t_vec[i + 1] - t_vec[i]) * f(t_vec[i], y[i])
    return y


def linear_interpol(t, t_vec, y_vec):
    y = print("damn")
    return y
