import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, Slider
from rkh import solve_dde_rk4_hermite

# --- Model Parameters (ALL will be global/slider-controlled) ---
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

# Initial values for h and t_end sliders (as requested)
t_end_INIT = 500.0 
h_step_INIT = 0.05 

# --- DDE System Definition (uses all global parameters for f_dde and alpha_func) ---
def f_dde_svir(t, y, y_delayed):
    S, V, I, R = y
    S_tau, V_tau, I_tau, R_tau = y_delayed 
    N_current = S + V + I + R
    if N_current < 1e-6: N_current = 1e-6 

    # Access current global parameter values for this f_dde call
    dSdt = _LAMBDA + (1 - _P) * _A - (_BETA * S * I) / N_current - _MU * S - _PSI * S_tau + _THETA * V
    dVdt = _PSI * S_tau - (_SIGMA * _BETA * V * I) / N_current - (_MU + _THETA) * V
    dIdt = _P * _A + (_BETA * S * I) / N_current + (_SIGMA * _BETA * V * I) / N_current - (_MU + _ALPHA) * I
    dRdt = _ALPHA * I - _MU * R
    return np.array([dSdt, dVdt, dIdt, dRdt])

def alpha_func_svir(t):
    return t - _TAU

# History function Y(t) for t <= t_start (these are fixed initial values for populations)
S0_HIST = 900.0
V0_HIST = 100.0
I0_HIST = 1.0
R0_HIST = 0.0
def phi_func_svir(t):
    return np.array([S0_HIST, V0_HIST, I0_HIST, R0_HIST])

# DDE Solver Parameters (initial for plot, overridden by sliders)
t_start_svir = 0.0
y_initial_svir = phi_func_svir(t_start_svir) 

# --- Function to solve the DDE for given parameters ---
# This function now takes ALL parameters as arguments
def solve_svir_for_plot(
    lambda_val, a_val, mu_val, p_val, alpha_val_param, theta_val, sigma_val, # All model parameters
    beta_val, psi_val, tau_val, t_end_val, h_step_val # Interactive parameters
): 
    # Declare all global parameters that are updated by sliders
    global _LAMBDA, _A, _MU, _P, _ALPHA, _THETA, _SIGMA, _BETA, _PSI, _TAU 
    
    # Update global variables with current slider values
    _LAMBDA = lambda_val
    _A = a_val
    _MU = mu_val
    _P = p_val
    _ALPHA = alpha_val_param # Renamed to avoid conflict with alpha_func
    _THETA = theta_val
    _SIGMA = sigma_val
    _BETA = beta_val
    _PSI = psi_val
    _TAU = tau_val

    # Basic input validation for solver parameters
    if h_step_val <= 0: h_step_val = 1e-6 
    if t_end_val <= t_start_svir: t_end_val = t_start_svir + h_step_val 

    # Call the DDE solver
    history_data = solve_dde_rk4_hermite(
        f_dde_svir,
        alpha_func_svir, # uses global _TAU
        phi_func_svir,
        (t_start_svir, t_end_val), 
        y_initial_svir,
        h_step_val, 
        h_disc_guess=0.01, # This can remain fixed
        constant_delay_value=_TAU # Pass _TAU for optimization
    )
    
    # Unpack history_data for plotting
    times = [item[0] for item in history_data]
    solutions = np.array([item[1] for item in history_data]) 

    return times, solutions[:, 0], solutions[:, 1], solutions[:, 2], solutions[:, 3] 

# --- Setup the Matplotlib figure and subplots ---
fig = plt.figure(figsize=(14, 8)) # Slightly taller figure for more sliders
ax_ts = fig.add_subplot(1, 2, 1) # Left subplot: Time series plot
ax_phase = fig.add_subplot(1, 2, 2, projection='3d') # Right subplot: Phase space plot (S,V,I)

# Initial solution calculation for the first plot display
initial_times, initial_S, initial_V, initial_I, initial_R = solve_svir_for_plot(
    _LAMBDA, _A, _MU, _P, _ALPHA, _THETA, _SIGMA, # All initial model parameters
    _BETA, _PSI, _TAU, t_end_INIT, h_step_INIT # Initial slider values for t_end, h_step
)

# Plot initial time series for S, V, I, R
line_S, = ax_ts.plot(initial_times, initial_S, label='S(t)', color='blue')
line_V, = ax_ts.plot(initial_times, initial_V, label='V(t)', color='red')
line_I, = ax_ts.plot(initial_times, initial_I, label='I(t)', color='green')
line_R, = ax_ts.plot(initial_times, initial_R, label='R(t)', color='purple')
ax_ts.set_xlabel("Time (t)")
ax_ts.set_ylabel("Population")
ax_ts.set_title("SVIR DDE Time Series")
ax_ts.legend(loc='upper right')
ax_ts.grid(True)
ax_ts.set_xlim(t_start_svir, t_end_INIT) # Set fixed x-limits initially for consistent scaling

# Plot initial phase space (S,V,I)
line_phase, = ax_phase.plot(initial_S, initial_V, initial_I, linewidth=1)
ax_phase.set_xlabel("S")
ax_phase.set_ylabel("V")
ax_phase.set_zlabel("I")
ax_phase.set_title("SVIR Phase Space (S,V,I)")


# --- Sliders setup ---
# Adjust bottom margin to accommodate many sliders
fig.subplots_adjust(left=0.1, bottom=0.55, right=0.9, top=0.9) 

# Define common slider height and spacing for many sliders
slider_height = 0.02
slider_spacing = 0.035 # Reduced spacing to fit more sliders

# Sliders for _BETA, _PSI, _TAU (adjust positions)
ax_beta = fig.add_axes([0.1, 0.50, 0.35, slider_height]) 
beta_slider = Slider(ax=ax_beta, label='Beta (β)', valmin=0.01, valmax=0.5, valinit=_BETA)

ax_psi = fig.add_axes([0.1, 0.50 - slider_spacing, 0.35, slider_height])
psi_slider = Slider(ax=ax_psi, label='Psi (ψ)', valmin=0.01, valmax=0.5, valinit=_PSI)

ax_tau = fig.add_axes([0.1, 0.50 - 2*slider_spacing, 0.35, slider_height])
tau_slider = Slider(ax=ax_tau, label='Tau (τ)', valmin=0.0, valmax=20.0, valinit=_TAU)

# NEW SLIDERS for the _FIXED parameters
# Lambda (Λ)
ax_lambda = fig.add_axes([0.1, 0.50 - 3*slider_spacing, 0.35, slider_height])
lambda_slider = Slider(ax=ax_lambda, label='Lambda (Λ)', valmin=0.0, valmax=1000.0, valinit=_LAMBDA, valstep=10.0)

# A
ax_A = fig.add_axes([0.55, 0.50, 0.35, slider_height]) # Start new column of sliders
A_slider = Slider(ax=ax_A, label='A', valmin=0.0, valmax=50.0, valinit=_A, valstep=1.0)

# Mu (μ)
ax_mu = fig.add_axes([0.55, 0.50 - slider_spacing, 0.35, slider_height])
mu_slider = Slider(ax=ax_mu, label='Mu (μ)', valmin=0.01, valmax=0.5, valinit=_MU)

# P
ax_P = fig.add_axes([0.55, 0.50 - 2*slider_spacing, 0.35, slider_height])
p_slider = Slider(ax=ax_P, label='P', valmin=0.0, valmax=1.0, valinit=_P)

# Alpha (α)
ax_alpha = fig.add_axes([0.55, 0.50 - 3*slider_spacing, 0.35, slider_height])
alpha_slider = Slider(ax=ax_alpha, label='Alpha (α)', valmin=0.01, valmax=0.5, valinit=_ALPHA)

# Theta (θ)
ax_theta = fig.add_axes([0.1, 0.50 - 4*slider_spacing, 0.35, slider_height]) # Continue in first column
theta_slider = Slider(ax=ax_theta, label='Theta (θ)', valmin=0.0, valmax=0.2, valinit=_THETA)

# Sigma (σ)
ax_sigma = fig.add_axes([0.55, 0.50 - 4*slider_spacing, 0.35, slider_height]) # Continue in second column
sigma_slider = Slider(ax=ax_sigma, label='Sigma (σ)', valmin=0.0, valmax=0.5, valinit=_SIGMA)


# Sliders for t_end and h_step (position adjusted)
ax_tend = fig.add_axes([0.1, 0.50 - 5*slider_spacing, 0.35, slider_height]) 
tend_slider = Slider(ax=ax_tend, label='t_end', valmin=10.0, valmax=1000.0, valinit=t_end_INIT, valstep=10.0)

ax_hstep = fig.add_axes([0.55, 0.50 - 5*slider_spacing, 0.35, slider_height]) 
hstep_slider = Slider(ax=ax_hstep, label='h_step', valmin=0.001, valmax=0.1, valinit=h_step_INIT, valstep=0.001)


# --- Function to display parameters in text box ---
# Position this text box relative to the figure, not axes, so it stays fixed
param_text_box = fig.text(
    0.02, 0.98, # Position in figure coords (top left)
    '', # Empty string initially
    transform=fig.transFigure, fontsize=9,
    verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5)
)

def update_param_text():
    # Construct the text string with all parameter values
    text_content = (
        f"Parameters:\n"
        f"  Λ = {lambda_slider.val:.0f}\n" 
        f"  A = {A_slider.val:.0f}\n"      
        f"  μ = {mu_slider.val:.3f}\n"     
        f"  p = {p_slider.val:.2f}\n"      
        f"  α = {alpha_slider.val:.3f}\n"   
        f"  θ = {theta_slider.val:.3f}\n"   
        f"  σ = {sigma_slider.val:.3f}\n"   
        f"  β = {beta_slider.val:.3f}\n" 
        f"  ψ = {psi_slider.val:.3f}\n"  
        f"  τ = {tau_slider.val:.3f}\n"  
        f"  t_end = {tend_slider.val:.0f}\n"
        f"  h_step = {hstep_slider.val:.4f}\n"
    )
    param_text_box.set_text(text_content)
    fig.canvas.draw_idle() # Redraw just the canvas to update text

# Initial display of parameters
update_param_text()


# --- Update function for sliders ---
def update(val):
    # Get current slider values
    current_lambda = lambda_slider.val
    current_A = A_slider.val
    current_mu = mu_slider.val
    current_p = p_slider.val
    current_alpha = alpha_slider.val
    current_theta = theta_slider.val
    current_sigma = sigma_slider.val
    current_beta = beta_slider.val
    current_psi = psi_slider.val
    current_tau = tau_slider.val
    current_tend = tend_slider.val 
    current_hstep = hstep_slider.val 

    # Recalculate solution
    times, S, V, I, R = solve_svir_for_plot(
        current_lambda, current_A, current_mu, current_p, current_alpha, current_theta, current_sigma, 
        current_beta, current_psi, current_tau, current_tend, current_hstep 
    )

    # Update time series plot data
    line_S.set_data(times, S)
    line_V.set_data(times, V)
    line_I.set_data(times, I)
    line_R.set_data(times, R)
    
    # Auto-scale y-axis based on new solution range
    min_pop = min(S.min(), V.min(), I.min(), R.min())
    max_pop = max(S.max(), V.max(), I.max(), R.max())
    ax_ts.set_ylim(min_pop - 0.05*abs(min_pop), max_pop + 0.05*abs(max_pop)) 
    ax_ts.set_xlim(times[0], times[-1]) # Update x-limits as t_end changes

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

    update_param_text() # Update parameter display
    fig.canvas.draw_idle() 

# Register update function with sliders
lambda_slider.on_changed(update)
A_slider.on_changed(update)
mu_slider.on_changed(update)
p_slider.on_changed(update)
alpha_slider.on_changed(update)
theta_slider.on_changed(update)
sigma_slider.on_changed(update)
beta_slider.on_changed(update)
psi_slider.on_changed(update)
tau_slider.on_changed(update)
tend_slider.on_changed(update) 
hstep_slider.on_changed(update) 

# --- Reset button ---
resetax = fig.add_axes([0.8, 0.025, 0.1, 0.04]) 
button = Button(resetax, 'Reset', hovercolor='0.975')

def reset(event):
    # Reset all sliders to their initial values
    lambda_slider.reset(); A_slider.reset(); mu_slider.reset(); p_slider.reset()
    alpha_slider.reset(); theta_slider.reset(); sigma_slider.reset()
    beta_slider.reset(); psi_slider.reset(); tau_slider.reset()
    tend_slider.reset(); hstep_slider.reset()

button.on_clicked(reset)

plt.show()
