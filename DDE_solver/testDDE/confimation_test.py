import numpy as np
# WARN: this one is mine


def f(t, y, x1, x2, x3, x4):
    return -x1 + x2 - x3*x4


def real_sol(t):
    if t < 0:
        return 1
    elif 0 <= t <= 1:
        return -t
    elif 1 < t <= 2:
        return (1/2) * t**2 - t - (1/2)
    elif 2 < t <= 3:
        return (-1/6) * t**3 + (1/2) * t**2 - (7/6)
    elif 3 < t <= 4:
        return (1/24) * t**4 - (1/6) * t**3 - (1/4) * t**2 + t - (19/24)
    elif 4 < t <= 5:
        return (-1/120) * t**5 + (1/6) * t**4 - (5/3) * t**3 + (109/12) * t**2 - 24 * t + (2689/120)
    else:
        return np.nan


def alpha(t, y):
    return [t-1,  t-2, t-3, t-4]


def alpha1(t, y):
    return t-1


def alpha2(t, y):
    return t-2


def alpha3(t, y):
    return t-3


def alpha4(t, y):
    return t-4


def rk4_step(f, tn, yn, h, alphas, real_sol):
    alpha1, alpha2, alpha3, alpha4 = alphas

    t1 = tn
    y1 = yn
    # Y11 = 1
    Y11 = real_sol(alpha1(t1, y1))
    Y12 = real_sol(alpha2(t1, y1))
    Y13 = real_sol(alpha3(t1, y1))
    Y14 = real_sol(alpha3(t1, y1))
    k1 = f(t1, y1, Y11, Y12, Y13, Y14)

    print(f't1 = {t1}')
    print(f'alpha1 = {alpha(t1, y1)}')
    print(f'Y1 = {Y11, Y12, Y13, Y13}')
    print(f'k1 = {k1}')

    t2 = tn + 0.5 * h
    y2 = yn + 0.5 * h * k1
    # Y21 = real_sol(alpha1(t2, y2))
    Y21 = 1
    Y22 = real_sol(alpha2(t2, y2))
    Y23 = real_sol(alpha3(t2, y2))
    Y24 = real_sol(alpha3(t2, y2))
    k2 = f(t2, y2, Y21, Y22, Y23, Y24)
    print(f't2 = {t2}')
    print(f'alpha2 = {alpha(t2, y2)}')
    print(f'Y2 = {Y21, Y22, Y23, Y23}')
    print(f'k2 = {k2}')

    t3 = tn + 0.5 * h
    y3 = yn + 0.5 * h * k2
    # Y31 = real_sol(alpha1(t3, y3))
    Y31 = 1
    Y32 = real_sol(alpha2(t3, y3))
    Y33 = real_sol(alpha3(t3, y3))
    Y34 = real_sol(alpha3(t3, y3))
    k3 = f(t3, y3, Y31, Y32, Y33, Y34)
    print(f't3 = {t3}')
    print(f'alpha3 = {alpha(t3, y3)}')
    print(f'Y3 = {Y31, Y32, Y33, Y34}')
    print(f'k3 = {k3}')

    t4 = tn + h
    y4 = yn + h * k2
    # WARN: this is a problem for t = 1
    # Y41 = 1  # left
    # Y41 = -1
    Y41 = real_sol(alpha1(t4, y4))
    # WARN: this is a problem for t = 2
    # Y42 = 1  # left
    # Y42 = 0  # right
    Y42 = real_sol(alpha2(t4, y4))
    Y43 = real_sol(alpha3(t4, y4))
    Y44 = real_sol(alpha3(t4, y4))
    k4 = f(t4, y4, Y41, Y42, Y43, Y44)
    print(f't4 = {t4}')
    print(f'alpha4 = {alpha(t4, y4)}')
    print(f'Y4 = {Y41, Y42, Y43, Y43}')
    print(f'k4 = {k4}')

    y_next = yn + h * (k1/6 + k2/3 + k3/3 + k4/6)

    return y_next


# tn = 0.95
# h = 0.05
# yn = real_sol(tn)
# alphas = [alpha1, alpha2, alpha3, alpha4]
# y1 = rk4_step(f, tn, yn, h, alphas, real_sol)
# y1_exact = real_sol(tn + h)


tn = 0.975
h = 0.05
yn = real_sol(tn)
alphas = [alpha1, alpha2, alpha3, alpha4]
y1 = rk4_step(f, tn, yn, h, alphas, real_sol)
y1_exact = real_sol(tn + h)


# tn = 1
# h = 0.05
# yn = real_sol(tn)
# alphas = [alpha1, alpha2, alpha3, alpha4]
# y1 = rk4_step(f, tn, yn, h, alphas, real_sol)
# y1_exact = real_sol(tn + h)


# tn = 1.95
# h = 0.05
# yn = real_sol(tn)
# alphas = [alpha1, alpha2, alpha3, alpha4]
# y1 = rk4_step(f, tn, yn, h, alphas, real_sol)
# y1_exact = real_sol(tn + h)

# tn = 2
# h = 0.05
# yn = real_sol(tn)
# alphas = [alpha1, alpha2, alpha3, alpha4]
# y1 = rk4_step(f, tn, yn, h, alphas, real_sol)
# y1_exact = real_sol(tn + h)


print(f'tn = {tn} h = {h}')
print('y1 =', y1)
print('y(t1)', y1_exact)
print(f'ERROR at t = {tn + h} is {abs(y1 - y1_exact)}')
print(f'{'_'*80}')
