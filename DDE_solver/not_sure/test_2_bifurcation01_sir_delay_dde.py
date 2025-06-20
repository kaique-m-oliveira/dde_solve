# test_dde_svir.py

import matplotlib.pyplot as plt
import numpy as np
from rkh import solve_dde_rk4_hermite

# --- Define the DDE components for the SVIR model ---

# Model Parameters (from problem description and paper)
_LAMBDA = 500.0  # Birth/Recruitment rate
_A = 25.0  # Vaccination rate
_PSI = 0.22  # Rate of vaccination from susceptible (delayed)
_BETA = 0.1  # Infection rate (S to I)
_MU = 0.1  # Natural death rate
_P = 0.1  # Proportion of A that becomes infected (direct infection)
_ALPHA = 0.2  # Recovery rate from I to R
_THETA = 0.05  # Vaccine-induced immunity waning rate (V to S)
_SIGMA = 0.1  # Factor for vaccine-breakthrough infection (V to I)

# --- CHANGE THIS LINE to match the paper's bifurcation point ---
_TAU = 14.3  # Delay in vaccination (critical delay for Hopf bifurcation)
# --- END CHANGE ---


# Function defining the DDE system
def f_dde_svir(t, y, y_delayed):
    S, V, I, R = y
    S_tau, V_tau, I_tau, R_tau = y_delayed

    N_current = S + V + I + R
    if N_current < 1e-6:
        N_current = 1e-6

    dSdt = (
        _LAMBDA
        + (1 - _P) * _A
        - (_BETA * S * I) / N_current
        - _MU * S
        - _PSI * S_tau
        + _THETA * V
    )
    dVdt = _PSI * S_tau - (_SIGMA * _BETA * V * I) / N_current - (_MU + _THETA) * V
    dIdt = (
        _P * _A
        + (_BETA * S * I) / N_current
        + (_SIGMA * _BETA * V * I) / N_current
        - (_MU + _ALPHA) * I
    )
    dRdt = _ALPHA * I - _MU * R

    return np.array([dSdt, dVdt, dIdt, dRdt])


# Delay function: alpha(t) = t - tau
def alpha_func_svir(t):
    return t - _TAU


# History function Y(t) for t <= t_start (constant values for simplicity)
def phi_func_svir(t):
    S0_hist = 900.0
    V0_hist = 100.0
    I0_hist = 1.0
    R0_hist = 0.0
    return np.array([S0_hist, V0_hist, I0_hist, R0_hist])


analytical_sol_func_svir = None

# --- DDE Solver Parameters ---
t_start_svir = 0.0
# --- CHANGE THIS LINE to match the paper's plot duration ---
t_end_svir = 1000.0  # Run for a longer period to observe oscillations
# --- END CHANGE ---

y_initial_svir = np.array(
    [
        phi_func_svir(t_start_svir)[0],
        phi_func_svir(t_start_svir)[1],
        phi_func_svir(t_start_svir)[2],
        phi_func_svir(t_start_svir)[3],
    ]
)

# Step size should be small enough for accuracy and stability with oscillations
h_step_svir = 0.05
h_disc_guess_svir = 0.01

# --- Main execution block ---
if __name__ == "__main__":
    print(f"--- Running Test: SVIR DDE Model (Tau={_TAU}) ---")
    print(f"Integrating from {t_start_svir} to {t_end_svir} with h={h_step_svir}")

    history_data = solve_dde_rk4_hermite(
        f_dde_svir,
        alpha_func_svir,
        phi_func_svir,
        (t_start_svir, t_end_svir),
        y_initial_svir,
        h_step_svir,
        h_disc_guess_svir,
    )

    times = [item[0] for item in history_data]
    solutions = np.array([item[1] for item in history_data])

    print(f"  Solver completed. Total steps: {len(times)}")

    if np.any(solutions < -1e-6):
        print("  Test FAILED: Population component(s) became negative.")
    else:
        print("  Test PASSED: Population components remained non-negative.")

    total_population = solutions.sum(axis=1)
    print(f"  Initial Total Population: {total_population[0]:.2f}")
    print(f"  Final Total Population: {total_population[-1]:.2f}")

    # --- Plotting Results ---
    plt.figure(figsize=(12, 8))

    plt.plot(times, solutions[:, 0], label="S(t) - Susceptible")
    plt.plot(times, solutions[:, 1], label="V(t) - Vaccinated")
    plt.plot(times, solutions[:, 2], label="I(t) - Infected")
    plt.plot(times, solutions[:, 3], label="R(t) - Recovered")
    plt.plot(
        times,
        total_population,
        label="N(t) - Total Population",
        linestyle="--",
        color="black",
    )

    plt.title(f"SVIR DDE Model Solution (Tau={_TAU})")
    plt.xlabel("Time (t)")
    plt.ylabel("Population")
    plt.legend()
    plt.grid(True)
    plt.yscale("linear")
    plt.savefig(f"dde_svir_solution_plot_tau_{_TAU}.png")  # Save with tau in filename
    plt.show()
