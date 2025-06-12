import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, Slider, TextBox

# --- FIX: Add the missing import for the DDE solver ---
from rkh import solve_dde_rk4_hermite

# --- END FIX ---


# --- Model Parameters (ALL will be global and controlled by sliders) ---
# Global variables for parameters controlled by sliders (initial values)
_LAMBDA = 500.0
_A = 25.0
_MU = 0.1
_P = 0.1
_ALPHA = 0.2
_THETA = 0.05
_SIGMA = 0.1
_BETA = 0.1
_PSI = 0.22
_TAU = 14.3

# Initial values for h and t_end sliders
t_end_INIT = 500.0
h_step_INIT = 0.05


# --- DDE System Definition ---
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


def alpha_func_svir(t):
    return t - _TAU


def phi_func_svir(t):
    S0_HIST = 900.0
    V0_HIST = 100.0
    I0_HIST = 1.0
    R0_HIST = 0.0
    return np.array([S0_HIST, V0_HIST, I0_HIST, R0_HIST])


# DDE Solver Parameters (initial for plot, overridden by sliders)
t_start_svir = 0.0
y_initial_svir = phi_func_svir(t_start_svir)


# --- Function to solve the DDE for given parameters ---
def solve_svir_for_plot(
    lambda_val,
    a_val,
    mu_val,
    p_val,
    alpha_val_param,
    theta_val,
    sigma_val,
    beta_val,
    psi_val,
    tau_val,
    t_end_val,
    h_step_val,
):
    # Declare all global parameters that are updated by sliders
    global _LAMBDA, _A, _MU, _P, _ALPHA, _THETA, _SIGMA, _BETA, _PSI, _TAU

    # Update global variables with current slider values
    _LAMBDA = lambda_val
    _A = a_val
    _MU = mu_val
    _P = p_val
    _ALPHA = alpha_val_param
    _THETA = theta_val
    _SIGMA = sigma_val
    _BETA = beta_val
    _PSI = psi_val
    _TAU = tau_val

    # Basic input validation for solver parameters
    if h_step_val <= 0:
        h_step_val = 1e-6
    if t_end_val <= t_start_svir:
        t_end_val = t_start_svir + h_step_val

    history_data = solve_dde_rk4_hermite(
        f_dde_svir,
        alpha_func_svir,
        phi_func_svir,
        (t_start_svir, t_end_val),
        y_initial_svir,
        h_step_val,
        h_disc_guess=0.01,
        constant_delay_value=_TAU,
    )

    times = [item[0] for item in history_data]
    solutions = np.array([item[1] for item in history_data])

    return times, solutions[:, 0], solutions[:, 1], solutions[:, 2], solutions[:, 3]


# --- Setup the Matplotlib figure and subplots ---

fig = plt.figure(figsize=(14, 8))
ax_ts = fig.add_subplot(1, 2, 1)
ax_phase = fig.add_subplot(1, 2, 2, projection="3d")


# Initial solution calculation
initial_times, initial_S, initial_V, initial_I, initial_R = solve_svir_for_plot(
    _LAMBDA,
    _A,
    _MU,
    _P,
    _ALPHA,
    _THETA,
    _SIGMA,
    _BETA,
    _PSI,
    _TAU,
    t_end_INIT,
    h_step_INIT,
)

# Plot initial time series
(line_S,) = ax_ts.plot(initial_times, initial_S, label="S(t)", color="blue")
(line_V,) = ax_ts.plot(initial_times, initial_V, label="V(t)", color="red")
(line_I,) = ax_ts.plot(initial_times, initial_I, label="I(t)", color="green")
(line_R,) = ax_ts.plot(initial_times, initial_R, label="R(t)", color="purple")
ax_ts.set_xlabel("Time (t)")
ax_ts.set_ylabel("Population")
ax_ts.set_title("SVIR DDE Time Series")
ax_ts.legend(loc="upper right")
ax_ts.grid(True)
ax_ts.set_xlim(t_start_svir, t_end_INIT)

# Plot initial phase space (S,V,I)
(line_phase,) = ax_phase.plot(initial_S, initial_V, initial_I, linewidth=1)
ax_phase.set_xlabel("S")
ax_phase.set_ylabel("V")
ax_phase.set_zlabel("I")
ax_phase.set_title("SVIR Phase Space (S,V,I)")


# --- Sliders setup ---

fig.subplots_adjust(left=0.1, bottom=0.4, right=0.9, top=0.9)
# WARN: a bit too small
# fig.subplots_adjust(left=0.1, bottom=0.60, right=0.9, top=0.9)

slider_height = 0.02
slider_spacing = 0.035

current_bottom_pos = 0.3

# List of parameter configurations (label, initial_value, min, max, step, valfmt)
param_configs = [
    ("Lambda (Λ)", _LAMBDA, 0.0, 1000.0, 10.0, "%0.0f"),  # FIXED valfmt
    ("A", _A, 0.0, 50.0, 1.0, "%0.0f"),  # FIXED valfmt
    ("Mu (μ)", _MU, 0.01, 0.5, 0.01, "%0.3f"),  # FIXED valfmt
    ("P", _P, 0.0, 1.0, 0.01, "%0.2f"),  # FIXED valfmt
    ("Alpha (α)", _ALPHA, 0.01, 0.5, 0.01, "%0.3f"),  # FIXED valfmt
    ("Theta (θ)", _THETA, 0.0, 0.2, 0.005, "%0.3f"),  # FIXED valfmt
    ("Sigma (σ)", _SIGMA, 0.0, 0.5, 0.01, "%0.3f"),  # FIXED valfmt
    ("Beta (β)", _BETA, 0.01, 0.5, 0.01, "%0.3f"),  # FIXED valfmt
    ("Psi (ψ)", _PSI, 0.01, 0.5, 0.01, "%0.3f"),  # FIXED valfmt
    ("Tau (τ)", _TAU, 0.0, 20.0, 0.1, "%0.3f"),  # FIXED valfmt
    ("t_end", t_end_INIT, 10.0, 1000.0, 10.0, "%0.0f"),  # FIXED valfmt
    ("h_step", h_step_INIT, 0.001, 0.1, 0.001, "%0.4f"),  # FIXED valfmt
]

# Lists to store slider and textbox objects
sliders = []
text_boxes = []

# Create sliders and text boxes dynamically
for i, (label, init_val, vmin, vmax, vstep, fmt) in enumerate(param_configs):
    col = 0 if i < 6 else 1  # Two columns of sliders
    row_in_col = i % 6  # Row index within its column

    slider_ax_left = 0.1 if col == 0 else 0.55  # X-position for slider axes
    textbox_ax_left = (
        slider_ax_left + 0.35 + 0.01
    )  # X-position for textbox axes (right of slider)

    current_row_bottom = current_bottom_pos - row_in_col * slider_spacing

    # Create slider
    ax_slider = fig.add_axes([slider_ax_left, current_row_bottom, 0.35, slider_height])
    slider = Slider(
        ax=ax_slider,
        label=label,
        valmin=vmin,
        valmax=vmax,
        valinit=init_val,
        valstep=vstep,
        valfmt=fmt,
    )
    sliders.append(slider)

    # Create text box
    ax_textbox = fig.add_axes(
        [textbox_ax_left - 0.005, current_row_bottom, 0.03, slider_height]
    )  # Changed width
    textbox = TextBox(ax_textbox, "", initial=fmt % init_val)
    text_boxes.append(textbox)

    # WARN: these textboxes were too big
    # ax_textbox = fig.add_axes(
    #     [textbox_ax_left, current_row_bottom, 0.08, slider_height]
    # )
    # textbox = TextBox(ax_textbox, "", initial=fmt % init_val)  # Use % for initial text
    # text_boxes.append(textbox)

    # --- Function to display parameters in text box ---
    param_text_box = fig.text(
        0.90,
        0.98,
        "",
        transform=fig.transFigure,
        fontsize=9,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.5", fc="wheat", alpha=0.5),
    )


def update_param_text():
    # Retrieve values from all sliders and format them using their original valfmt
    text_content = "Parameters:\n"
    for i, (label, _, _, _, _, fmt) in enumerate(param_configs):
        text_content += f"  {label} = {fmt % sliders[i].val}\n"  # Use % for formatting

    param_text_box.set_text(text_content)
    fig.canvas.draw_idle()


# Initial display of parameters
update_param_text()


# --- Update function for sliders ---
def update(val):  # 'val' argument is from on_changed callback, not used directly here
    # Get current slider values directly
    current_params = [s.val for s in sliders]

    # Unpack into named variables for solve_svir_for_plot
    (
        current_lambda,
        current_A,
        current_mu,
        current_p,
        current_alpha,
        current_theta,
        current_sigma,
        current_beta,
        current_psi,
        current_tau,
        current_tend,
        current_hstep,
    ) = current_params

    # Recalculate solution
    times, S, V, I, R = solve_svir_for_plot(
        current_lambda,
        current_A,
        current_mu,
        current_p,
        current_alpha,
        current_theta,
        current_sigma,
        current_beta,
        current_psi,
        current_tau,
        current_tend,
        current_hstep,
    )

    # Update time series plot data
    line_S.set_data(times, S)
    line_V.set_data(times, V)
    line_I.set_data(times, I)
    line_R.set_data(times, R)

    # Auto-scale y-axis based on new solution range
    min_pop = min(S.min(), V.min(), I.min(), R.min())
    max_pop = max(S.max(), V.max(), I.max(), R.max())
    ax_ts.set_ylim(min_pop - 0.05 * abs(min_pop), max_pop + 0.05 * abs(max_pop))
    ax_ts.set_xlim(times[0], times[-1])

    # Clear and redraw phase space plot
    ax_phase.clear()
    ax_phase.plot(S, V, I, linewidth=1)
    ax_phase.set_xlabel("S")
    ax_phase.set_ylabel("V")
    ax_phase.set_zlabel("I")
    ax_phase.set_title("SVIR Phase Space (S,V,I)")

    # Auto-scale phase space axes
    ax_phase.set_xlim(S.min(), S.max())
    ax_phase.set_ylim(V.min(), V.max())
    ax_phase.set_zlim(I.min(), I.max())

    update_param_text()
    fig.canvas.draw_idle()


# --- Setup text box submission ---
def submit_text_value(
    text_val, slider_obj, textbox_obj, param_idx
):  # param_idx added for fmt lookup
    try:
        val = float(text_val)
        # Clip value to slider's range
        val = max(slider_obj.valmin, min(val, slider_obj.valmax))
        slider_obj.set_val(val)  # This triggers slider's on_changed callback (update)
        # Update textbox to show clipped/formatted value
        textbox_obj.set_val(param_configs[param_idx][5] % val)  # Use % for formatting
    except ValueError:
        # If input is not a valid number, reset textbox to current slider value
        textbox_obj.set_val(param_configs[param_idx][5] % slider_obj.val)
    fig.canvas.draw_idle()


# Register update function with sliders and submission function with textboxes
for i, (label, init_val, vmin, vmax, vstep, fmt) in enumerate(param_configs):
    slider = sliders[i]
    textbox = text_boxes[i]

    slider.on_changed(update)
    # Pass the index 'i' to submit_text_value for fmt lookup
    textbox.on_submit(
        lambda text, s=slider, tb=textbox, idx=i: submit_text_value(text, s, tb, idx)
    )

# --- Reset button ---
resetax = fig.add_axes([0.8, 0.025, 0.1, 0.04])
button = Button(resetax, "Reset", hovercolor="0.975")


def reset(event):
    for i, (label, init_val, vmin, vmax, vstep, fmt) in enumerate(param_configs):
        slider = sliders[i]
        textbox = text_boxes[i]
        slider.reset()  # This triggers on_changed, which calls update
        # Reset the textbox value explicitly after slider reset
        textbox.set_val(fmt % slider.val)  # Use % for formatting
    fig.canvas.draw_idle()


button.on_clicked(reset)

plt.show()
