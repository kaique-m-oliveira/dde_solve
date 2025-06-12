import matplotlib  # For setting the backend
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, Slider
from rkh import solve_dde_rk4_hermite  # Assuming your DDE solver is in rkh.py

# Set a non-interactive backend first if not running from interactive environment,
# or for explicit interactive display. For Linux terminal, TkAgg is often necessary.
# matplotlib.use("TkAgg")

# --- Define the DDE components for the SVIR model ---

# Model Parameters (as from paper, but _BETA, _PSI, _TAU will be dynamic via sliders)
_LAMBDA_FIXED = 500.0
_A_FIXED = 25.0
_MU_FIXED = 0.1
_P_FIXED = 0.1
_ALPHA_FIXED = 0.2
_THETA_FIXED = 0.05
_SIGMA_FIXED = 0.1

# Define initial values for sliders and make them global for f_dde_svir and alpha_func_svir
_BETA = 0.1  # Initial beta value
_PSI = 0.22  # Initial psi value
_TAU = 14.3  # Initial tau value (critical delay for Hopf bifurcation from paper)


# DDE system function. This will use the current global values of _BETA, _PSI, _TAU.
def f_dde_svir(t, y, y_delayed):
    S, V, I, R = y
    # y_delayed will be [S(t-tau), V(t-tau), I(t-tau), R(t-tau)]
    S_tau, V_tau, I_tau, R_tau = y_delayed

    N_current = S + V + I + R
    if N_current < 1e-6:
        N_current = 1e-6  # Avoid division by zero

    dSdt = (
        _LAMBDA_FIXED
        + (1 - _P_FIXED) * _A_FIXED
        - (_BETA * S * I) / N_current
        - _MU_FIXED * S
        - _PSI * S_tau
        + _THETA_FIXED * V
    )
    dVdt = (
        _PSI * S_tau
        - (_SIGMA_FIXED * _BETA * V * I) / N_current
        - (_MU_FIXED + _THETA_FIXED) * V
    )
    dIdt = (
        _P_FIXED * _A_FIXED
        + (_BETA * S * I) / N_current
        + (_SIGMA_FIXED * _BETA * V * I) / N_current
        - (_MU_FIXED + _ALPHA_FIXED) * I
    )
    dRdt = _ALPHA_FIXED * I - _MU_FIXED * R

    return np.array([dSdt, dVdt, dIdt, dRdt])


# Delay function: alpha(t) = t - _TAU (uses global _TAU)
def alpha_func_svir(t):
    return t - _TAU


# History function Y(t) for t <= t_start (constant initial values)
S0_HIST = 900.0
V0_HIST = 100.0
I0_HIST = 1.0
R0_HIST = 0.0


def phi_func_svir(t):
    return np.array([S0_HIST, V0_HIST, I0_HIST, R0_HIST])


# DDE Solver Parameters (adjusted for interactivity - shorter time, larger steps)
t_start_svir = 0.0
t_end_svir = 200.0  # Shorter time span for faster updates
y_initial_svir = np.array([S0_HIST, V0_HIST, I0_HIST, R0_HIST])
h_step_svir = 0.5  # Larger step size for faster updates
h_disc_guess_svir = 0.01  # Heuristic for discontinuity finder


# --- Function to solve the DDE for given parameters ---
def solve_svir_for_plot(beta_val, psi_val, tau_val):
    global _BETA, _PSI, _TAU  # Declare globals to modify them
    _BETA = beta_val
    _PSI = psi_val
    _TAU = tau_val  # Update global tau

    # Call the DDE solver. Pass _TAU for constant delay optimization.
    history_data = solve_dde_rk4_hermite(
        f_dde_svir,
        alpha_func_svir,
        phi_func_svir,
        (t_start_svir, t_end_svir),
        y_initial_svir,
        h_step_svir,
        h_disc_guess_svir,
        constant_delay_value=_TAU,
    )

    # Unpack for plotting
    times = [item[0] for item in history_data]
    solutions = np.array(
        [item[1] for item in history_data]
    )  # Convert to array for easy component access

    return (
        times,
        solutions[:, 0],
        solutions[:, 1],
        solutions[:, 2],
        solutions[:, 3],
    )  # Return S, V, I, R


# --- Setup the Matplotlib figure and subplots ---
fig = plt.figure(figsize=(14, 7))  # Adjust figure size for 2 plots + sliders
ax_ts = fig.add_subplot(1, 2, 1)  # Left subplot: Time series plot
ax_phase = fig.add_subplot(
    1, 2, 2, projection="3d"
)  # Right subplot: Phase space plot (S,V,I)

# Initial solution calculation for the first plot
initial_times, initial_S, initial_V, initial_I, initial_R = solve_svir_for_plot(
    _BETA, _PSI, _TAU  # Use current global initial values
)

# Plot initial time series for S, V, I, R
(line_S,) = ax_ts.plot(
    initial_times, initial_S, label="S(t) - Susceptible", color="blue"
)
(line_V,) = ax_ts.plot(initial_times, initial_V, label="V(t) - Vaccinated", color="red")
(line_I,) = ax_ts.plot(initial_times, initial_I, label="I(t) - Infected", color="green")
(line_R,) = ax_ts.plot(
    initial_times, initial_R, label="R(t) - Recovered", color="purple"
)
ax_ts.set_xlabel("Time (t)")
ax_ts.set_ylabel("Population")
ax_ts.set_title("SVIR DDE Time Series")
ax_ts.legend(loc="upper right")
ax_ts.grid(True)
ax_ts.set_xlim(t_start_svir, t_end_svir)  # Set fixed x-limits for consistent scaling

# Plot initial phase space (S,V,I)
(line_phase,) = ax_phase.plot(initial_S, initial_V, initial_I, linewidth=1)
ax_phase.set_xlabel("S")
ax_phase.set_ylabel("V")
ax_phase.set_zlabel("I")
ax_phase.set_title("SVIR Phase Space (S,V,I)")


# --- Sliders setup ---
# Make space for sliders at the bottom
fig.subplots_adjust(left=0.1, bottom=0.35, right=0.9, top=0.9)

ax_beta = fig.add_axes([0.1, 0.25, 0.35, 0.03])  # [left, bottom, width, height]
beta_slider = Slider(
    ax=ax_beta,
    label="Beta (β)",
    valmin=0.01,
    valmax=0.5,
    valinit=_BETA,  # Adjusted range for β
)

ax_psi = fig.add_axes([0.1, 0.20, 0.35, 0.03])
psi_slider = Slider(
    ax=ax_psi,
    label="Psi (ψ)",
    valmin=0.01,
    valmax=0.5,
    valinit=_PSI,  # Adjusted range for ψ
)

ax_tau = fig.add_axes([0.1, 0.15, 0.35, 0.03])
tau_slider = Slider(
    ax=ax_tau,
    label="Tau (τ)",
    valmin=0.0,
    valmax=20.0,
    valinit=_TAU,  # Range to show bifurcation
)


# --- Update function for sliders ---
def update(val):
    # Get current slider values
    current_beta = beta_slider.val
    current_psi = psi_slider.val
    current_tau = tau_slider.val

    # Recalculate solution
    times, S, V, I, R = solve_svir_for_plot(current_beta, current_psi, current_tau)

    # Update time series plot data
    line_S.set_data(times, S)
    line_V.set_data(times, V)
    line_I.set_data(times, I)
    line_R.set_data(times, R)

    # Auto-scale y-axis based on new solution range
    min_pop = min(S.min(), V.min(), I.min(), R.min())
    max_pop = max(S.max(), V.max(), I.max(), R.max())
    ax_ts.set_ylim(
        min_pop - 0.05 * abs(min_pop), max_pop + 0.05 * abs(max_pop)
    )  # Add padding

    # Clear and redraw phase space plot
    ax_phase.clear()  # Clear existing lines and labels
    ax_phase.plot(S, V, I, linewidth=1)
    ax_phase.set_xlabel("S")  # Redraw labels after clear
    ax_phase.set_ylabel("V")
    ax_phase.set_zlabel("I")
    ax_phase.set_title("SVIR Phase Space (S,V,I)")

    # Auto-scale phase space axes
    ax_phase.set_xlim(S.min(), S.max())
    ax_phase.set_ylim(V.min(), V.max())
    ax_phase.set_zlim(I.min(), I.max())

    fig.canvas.draw_idle()  # Redraw the figure (more efficient than draw_idle())


# Register update function with sliders
beta_slider.on_changed(update)
psi_slider.on_changed(update)
tau_slider.on_changed(update)

# --- Reset button ---
resetax = fig.add_axes(
    [0.8, 0.025, 0.1, 0.04]
)  # Position: [left, bottom, width, height]
button = Button(resetax, "Reset", hovercolor="0.975")


def reset(event):
    beta_slider.reset()
    psi_slider.reset()
    tau_slider.reset()
    # Sliders' reset method automatically triggers on_changed callback, so no need to call update() explicitly.


button.on_clicked(reset)

plt.show()
