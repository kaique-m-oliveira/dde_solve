import numpy as np
from scipy.optimize import root
from rkh import find_discontinuity_chain  # Assuming your function is in this file

# --- Define Vector Alpha Functions ---


# Example 4: alpha(t) = [t-1, 0.5*t]
# For a given t_k, next root is min(t_k+1, 2*t_k)
def alpha_vector_mixed(t):
    return np.array([t - 1.0, 0.5 * t])


# Example 5: alpha(t) = [sqrt(t), t/2]
# For a given t_k, next root is min(t_k^2, 2*t_k)
# (assuming t_k > 0 for sqrt(t))
def alpha_vector_sqrt_linear(t):
    return np.array([np.sqrt(t), t / 2.0])


# --- Test Cases for Vector Alpha Functions ---

print("\n--- Running Discontinuity Chain Tests for Vector Alpha Functions ---")

# Test 4: Mixed vector delays
t0_ex4 = 0.5
tend_ex4 = 5.0
h_guess_ex4 = 0.1  # Small guess for robustness
discs_ex4 = find_discontinuity_chain(alpha_vector_mixed, t0_ex4, tend_ex4, h_guess_ex4)
print(f"\nTest 4 (alpha(t)=[t-1, 0.5t], t0={t0_ex4}, tf={tend_ex4}):")
print(f"  Found: {discs_ex4}")
print(f"  Expected: [0.5, 1.0, 2.0, 3.0, 4.0, 5.0]")

# Test 5: Square root and linear vector delays
t0_ex5 = 2.0
tend_ex5 = 30.0
h_guess_ex5 = 0.1  # Small guess for robustness
discs_ex5 = find_discontinuity_chain(
    alpha_vector_sqrt_linear, t0_ex5, tend_ex5, h_guess_ex5
)
print(f"\nTest 5 (alpha(t)=[sqrt(t), t/2], t0={t0_ex5}, tf={tend_ex5}):")
print(f"  Found: {discs_ex5}")
print(f"  Expected: [2.0, 4.0, 8.0, 16.0]")

print("\n--- Tests Finished ---")
