# DDE_solve

Numerical solver for Delay Differential Equations (DDEs) and Neutral Delay Differential Equations (NDDEs).

## Problem classes solved

### Delay differential equations (DDEs)

$$
\begin{cases}
y'(t) = f\bigl(t, y(t), y(\alpha_1(t,y(t))), \dots, y(\alpha_r(t,y(t)))\bigr), \quad t \ge t_0, \\
y(t) = \phi(t), \quad t \le t_0 .
\end{cases}
$$

All variables may be vector-valued in $\mathbb{R}^d$.  
Each delay satisfies $\alpha_i(t,y(t)) \le t$.

---

### Neutral delay differential equations (NDDEs)

$$
\begin{cases}
y'(t) = f\bigl(t, y(t), y(\alpha_1(t,y(t))), \dots, y(\alpha_r(t,y(t))), \\
\qquad\qquad y'(\beta_1(t,y(t))), \dots, y'(\beta_s(t,y(t)))\bigr), \quad t \ge t_0, \\
y(t) = \phi(t), \quad t \le t_0 .
\end{cases}
$$

Again the delays satisfy $\alpha_i(t,y(t)) \le t$ and $\beta_i(t,y(t)) \le t$.

---

# Example usage

Below are examples from the classical DDETST problems.

---

## Example 1 — DDE (scalar)

**B1 (Neves, 1975)**

$$
\begin{aligned}
y'(t) &= 1 - y(\exp(1 - 1/t)), \\
\phi(t) &= \ln(t), \quad 0 < t \le 0.1, \\
t_0 &= 0.1, \quad t_f = 10.
\end{aligned}
$$

Analytical solution:  
$$y(t) = \ln(t), \qquad \text{vanishing delay at } t = 1.$$

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
# plot(solution.t, solution.y)
````

---

## Example 2 — System of DDEs

**D1 (Neves, 1975)**

$$
\begin{aligned}
y_1'(t) &= y_2(t), \\
y_2'(t) &= - y_2(\exp(1 - y_2(t))) , y_2(t)^2 , e^{1 - y_2(t)}, \\
\phi_1(t) &= \ln(t), \quad \phi_2(t) = 1/t, \\
t_0 &= 0.1, \quad t_f = 5.
\end{aligned}
$$

Analytical solution:
$$y_1(t)=\ln(t), \qquad y_2(t)=1/t.$$

```python
import numpy as np
from dde_solve import solve_dde

def f(t, y, x):
    y1, y2 = y
    x1, x2 = x
    return [y2, -x2 * (y2**2) * np.exp(1 - y2)]

def phi(t):
    return [np.log(t), 1/t]

def alpha(t, y):
    y1, y2 = y
    return np.exp(1 - y2)

t_span = [0.1, 5]
solution = solve_dde(t_span, f, alpha, phi)
# plot(solution.t, solution.y)
```

---

## Example 3 — NDDE

**H2 (Hayashi, 1996)**

$$
\begin{aligned}
y'(t) &= \cos(t)\bigl(1 + y(t y^2(t))\bigr)
L_3 y(t) y'(t y^2(t)) \\
        &\quad + (1-L_3)\sin(t)\cos(t\sin^2 t) - \sin(t + t\sin^2 t), \\
        t_0 &= 0, \quad t_f = \pi, \\
        \phi(0) &= 0, \quad \phi'(t) = 1.
        \end{aligned}
        $$

Analytical solution:
$$y(t) = \sin(t).$$
Vanishing delays at $t = 0, \pi/2, \pi$.

```python
import numpy as np
from dde_solve import solve_ndde

L3 = 0.1

def f(t, y, x, z):
    return (np.cos(t)*(1 + x)
            + L3*y*z
            + (1-L3)*np.sin(t)*np.cos(t*np.sin(t)**2)
            - np.sin(t + t*np.sin(t)**2))

def phi(t):
    return 0

def phi_t(t):
    return 1

def alpha(t, y):
    return t*(y**2)

beta = alpha

t_span = [0, np.pi]
solution = solve_ndde(t_span, f, alpha, beta, phi, phi_t)
# plot(solution.t, solution.y)
```

---

# Installation

pip install dde-solve

---

**Test Repository**: [kaique-m-oliveira/dde_solve_tests](https://github.com/kaique-m-oliveira/ddetst)

---

# License

MIT License.
See the `LICENSE` file for details.

