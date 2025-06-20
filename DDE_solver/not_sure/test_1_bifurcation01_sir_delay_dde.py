import matplotlib.pyplot as plt
import numpy as np
from rkh import solve_dde_rk4_hermite  # Assuming your DDE solver is in rkh.py

# --- Define the DDE components for the SVIR model ---

# Model Parameters (from problem description)
_LAMBDA = 500.0  # Birth/Recruitment rate
_A = 25.0  # Vaccination rate (proportion of new births `pA` directly infected, `(1-p)A` become S)
_PSI = 0.22  # Rate of vaccination of susceptibles (psi*S(t-tau))
_BETA = 0.1  # Transmission rate (beta*S*I/N)
_MU = 0.1  # Natural death rate
_P = 0.1  # Proportion of `A` that becomes infected (p*A)
_ALPHA = 0.2  # Recovery rate from I to R
_THETA = 0.05  # Vaccine-induced immunity waning rate (V to S)
_SIGMA = 0.1  # Factor for vaccine-breakthrough infection (sigma*beta*V*I/N)

_TAU = 14.3  # Delay in vaccination (critical delay for Hopf bifurcation)
# --- END CHANGE ---
# _TAU = 0.5  # Delay in vaccination from S(t-tau) term (assumed constant here)


# Function defining the DDE system: y'(t) = f(t, y(t), y(alpha(t)))
# y = [S, V, I, R]
# y_delayed = [S(t-tau), V(t-tau), I(t-tau), R(t-tau)]
def f_dde_svir(t, y, y_delayed):
    S, V, I, R = y
    S_tau, V_tau, I_tau, R_tau = y_delayed  # Access components of delayed state

    # Current total population. Add a small epsilon to avoid division by zero if N becomes zero.
    N_current = S + V + I + R
    if N_current < 1e-6:
        N_current = 1e-6

    # Differential equations
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


# History function Y(t) for t <= t_start.
# We assume initial conditions (S0, V0, I0, R0) are constant over [-tau, 0].
def phi_func_svir(t):
    # These values should match y_initial_svir in solver parameters
    S0_hist = 900.0
    V0_hist = 100.0
    I0_hist = 1.0
    R0_hist = 0.0
    return np.array([S0_hist, V0_hist, I0_hist, R0_hist])


# No analytical solution provided for comparison
analytical_sol_func_svir = None

# --- DDE Solver Parameters ---
t_start_svir = 0.0
t_end_svir = 300.0  # Run for a longer period typical for epidemiological models
# Initial values for S(0), V(0), I(0), R(0)
y_initial_svir = np.array(
    [
        phi_func_svir(t_start_svir)[0],  # S(0)
        phi_func_svir(t_start_svir)[1],  # V(0)
        phi_func_svir(t_start_svir)[2],  # I(0)
        phi_func_svir(t_start_svir)[3],
    ]
)  # R(0)

# Use a small step size, as these models can be sensitive and potentially stiff
h_step_svir = 0.05
h_disc_guess_svir = 0.01  # Heuristic for discontinuity finder

# --- Main execution block ---
if __name__ == "__main__":
    print(f"--- Running Test: SVIR DDE Model ---")
    print(f"Integrating from {t_start_svir} to {t_end_svir} with h={h_step_svir}")

    # Solve the DDE
    history_data = solve_dde_rk4_hermite(
        f_dde_svir,
        alpha_func_svir,
        phi_func_svir,
        (t_start_svir, t_end_svir),
        y_initial_svir,
        h_step_svir,
        h_disc_guess_svir,
    )

    # Unpack times and solutions from history_data for analysis/plotting
    times = [item[0] for item in history_data]
    solutions = np.array(
        [item[1] for item in history_data]
    )  # Convert to array for easy component access

    print(f"  Solver completed. Total steps: {len(times)}")

    # --- Check for plausibility (since no analytical solution) ---
    # Check if any population component went negative
    if np.any(solutions < -1e-6):  # Allow tiny negative for float precision
        print("  Test FAILED: Population component(s) became negative.")
    else:
        print("  Test PASSED: Population components remained non-negative.")

    # Calculate and print total population dynamics
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

    plt.title("SVIR DDE Model Solution")
    plt.xlabel("Time (t)")
    plt.ylabel("Population")
    plt.legend()
    plt.grid(True)
    plt.yscale("linear")  # Ensure scale is linear, not log, unless specified
    plt.savefig("dde_svir_solution_plot.png")
    plt.show()
