import random

import matplotlib.pyplot as plt
import numpy as np
import scipy as sc
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicHermiteSpline, CubicSpline
from scipy.optimize import fsolve

t = np.arange(0, 10, 0.01)
sin = np.sin(t)
hsin = CubicSpline(t, sin)

hsin_list = [hsin(i) for i in t]

plt.plot(t, sin)
plt.plot(t, hsin_list)
plt.show()

error = 0
for i in range(100):
    x = random.uniform(0, 10)
    er = abs(np.sin(x) - hsin(x))
    print("x", x, "er", er)
    if er > error:
        error = er
print(error)
