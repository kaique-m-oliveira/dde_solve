
---

# **dde-solve**

A numerical solver for **delay differential equations (DDEs)** and **neutral delay differential equations (NDDEs)**.
The library supports state-dependent delays, breaking discontinuities, overlap handling, and adaptive continuous Runge–Kutta methods.

---

## **Mathematical formulation**

Numerical integrator for problems of the following kind.

### **Delay differential equation (DDE)**

[
\begin{cases}
y'(t)=f\bigl(t,,y(t),,y(\alpha_1(t,y(t))),\dots,y(\alpha_r(t,y(t)))\bigr), & t\ge t_0,[3pt]
y(t)=\phi(t), & t\le t_0,
\end{cases}
]

where (y,f,\phi:\mathbb{R}\to\mathbb{R}^d) and each delay (\alpha_i) satisfies

[
\alpha_i(t,y(t))\le t .
]

---

### **Neutral delay differential equation (NDDE)**

[
\begin{cases}
y'(t)=f\bigl(t,, y(t),, y(\alpha_1(t,y(t))),\dots, y(\alpha_r(t,y(t))), {}[3pt]
\qquad\qquad y'(\beta_1(t,y(t))),\dots,y'(\beta_s(t,y(t)))\bigr), & t\ge t_0,[3pt]
y(t)=\phi(t), & t\le t_0,
\end{cases}
]

with (y,f,\phi:\mathbb{R}\to\mathbb{R}^d) and delays satisfying (\alpha_i(t,y(t))\le t), (\beta_i(t,y(t))\le t).

---

## **Installation**

```
pip install dde-solve
```

---

## **Basic usage**

### **Example 1 – Scalar DDE**

**Problem B1 - DDETST (Neves, 1975)**

[
\begin{aligned}
y'(t) &= 1 - y(\exp(1 - 1/t)),\
t_0 &= 0.1,\quad t_f = 10,\
\phi(t) &= \ln(t), \qquad 0<t\le 0.1 .
\end{aligned}
]

Analytic solution: (y(t)=\ln(t)).
Vanishing delay at (t=1).

```python
import numpy as np
from dde_solve import solve_dde

def f(t, y, x):
    return 1 - x

def phi(t):
    return np.log(t)

def alpha(t, y):
    return np.exp(1 - 1/t)

t_span = [0.1, 10]
solution = solve_dde(t_span, f, alpha, phi)

# Example plotting:
import matplotlib.pyplot as plt
plt.plot(solution.t, solution.y)
plt.show()
```

---

### **Example 2 – System of DDEs**

**Problem D1 - DDETST (Neves, 1975)**

[
\begin{aligned}
y_1'(t)&= y_2(t),\
y_2'(t)&= -y_2(\exp(1-y_2(t))),y_2^2(t),\exp(1-y_2(t)),\
\phi_1(t)&=\ln(t),\quad
\phi_2(t)=1/t .
\end{aligned}
]

Analytic solution:
(y_1(t)=\ln(t)), (y_2(t)=1/t).
Vanishing delay at (t=1).

```python
import numpy as np
from dde_solve import solve_dde

def f(t, y, x):
    y1, y2 = y
    x1, x2 = x
    dy1 = y2
    dy2 = -x2 * (y2**2) * np.exp(1 - y2)
    return [dy1, dy2]

def phi(t):
    return [np.log(t), 1/t]

def alpha(t, y):
    y1, y2 = y
    return np.exp(1 - y2)

t_span = [0.1, 5]
solution = solve_dde(t_span, f, alpha, phi)
```

---

### **Example 3 – Neutral DDE**

**Problem H2 - DDETST (Hayashi, 1996)**

[
\begin{aligned}
y'(t) &= \cos(t)\bigl(1+y(t,y^2(t))\bigr)
+ L_3 y(t) y'(t,y^2(t))\
&\quad + (1-L_3)\sin(t)\cos(t\sin^2(t))
-\sin(t+t\sin^2(t)),
\end{aligned}
]

with
(L_3 = 0.1),
(\phi(0)=0), (\phi'(t)=1),
analytic solution (y(t)=\sin(t)).

```python
import numpy as np
from dde_solve import solve_ndde

L3 = 0.1
def f(t, y, x, z):
    return (
        np.cos(t)*(1 + x)
        + L3*y*z
        + (1 - L3)*np.sin(t)*np.cos(t*np.sin(t)**2)
        - np.sin(t + t*np.sin(t)**2)
    )

def phi(t):
    return 0

def phi_t(t):
    return 1

def alpha(t, y):
    return t * (y**2)

beta = alpha

t_span = [0, np.pi]
solution = solve_ndde(t_span, f, alpha, beta, phi, phi_t)
```

---

## **Features**

* State-dependent delays
* Neutral delays (y'(\beta(t,y)))
* Adaptive Runge–Kutta methods (explicit CRK)
* Overlapping-interval fixed-point iteration
* Detection and validation of breaking discontinuities
* Continuous output (y(t)) and (y'(t))
* Works for vector-valued systems

---

## **License**

MIT License (see `LICENSE` file).

---

