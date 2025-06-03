def f(t, y):
    return y(t - 1) + y(t - 2)


def get_delay_funcs(f):
    id = lambda t: t
    return lambda t: f(t, id)


delay = lambda t: f(t, lambda t: t)
for i in range(10):
    print(delay(i))


# delay = get_delay_funcs(f)
