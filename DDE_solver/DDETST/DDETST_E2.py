import numpy as np

from DDE_solver.rkh_refactor import *

# -------------------------------------------------------------------
# Problem E2: Logistic Gauss-type predator–prey system (Kuang, 1991)
# -------------------------------------------------------------------

# Parameters
alpha_ = 1.0 / 10.0   # α = 0.1
rho_   = 29.0 / 10.0  # ρ = 2.9
tau_   = 21.0 / 50.0  # τ = 0.42

# -------------------------------------------------------------------
# Right-hand side of the system
# -------------------------------------------------------------------
def f(t, y, x, z):
    """
    y : [y1(t), y2(t)]
    x : delayed states = [y1(t - τ), y2(t - τ)]
    z : delayed derivatives = [y1'(t - τ), y2'(t - τ)]
    Returns [y1'(t), y2'(t)].
    """
    y1, y2 = y
    x1, x2 = x
    z1, z2 = z

    dy1 = y1 * (1 - x1 - rho_ * z1) - (y2 * y1**2) / (y1**2 + 1)
    dy2 = y2 * ( (y1**2) / (y1**2 + 1) - alpha_ )
    return np.array([dy1, dy2])

# -------------------------------------------------------------------
# History (initial functions) for t ≤ 0
# -------------------------------------------------------------------
def phi(t):
    """History values y(t) = [φ₁(t), φ₂(t)] for t ≤ 0."""
    phi1 = 33.0 / 100.0 - t / 10.0
    phi2 = 111.0 / 50.0 + t / 10.0
    return np.array([phi1, phi2])

def phi_t(t):
    """Derivatives of history (φ′(t)) for t ≤ 0."""
    dphi1 = -1.0 / 10.0
    dphi2 =  1.0 / 10.0
    return np.array([dphi1, dphi2])

# -------------------------------------------------------------------
# Delay mapping
# -------------------------------------------------------------------
def alpha(t, y):
    """Return the delayed time(s)."""
    return t - tau_

beta = alpha  

# -------------------------------------------------------------------
# Time span
# -------------------------------------------------------------------
t_span = [0.0, 2.0]


methods = ['RKC3', 'RKC4', 'RKC5']
tolerances = [1e-2, 1e-4, 1e-6, 1e-8, 1e-10]


for Tol in tolerances:
    print('===========================================================')
    print(f'Tol = {Tol} \n')
    for method in methods:
        solution = solve_ndde(t_span, f, alpha, beta, phi, phi_t, method = method, Atol=Tol, Rtol=Tol)

        
        print(f'method = {method}')
        print('No analytical solution')
        print('steps: ', solution.steps)
        print('fails: ', solution.fails)
        print('feval: ', solution.feval)
        print('')
