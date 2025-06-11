import numpy as np
from disc_finder import find_discontinuity_chain  # Import your function

# --- Define Example Alpha Functions ---


# Example 1: alpha(t) = t - 1 (constant delay)
def alpha_const_delay(t):
    return t - 1.0


# Example 2: alpha(t) = 0.5 * t (linear delay)
def alpha_linear_delay(t):
    return 0.5 * t


# Example 3: alpha(t) = t^2 / 4 (non-linear delay)
def alpha_nonlinear_delay(t):
    return t**2 / 4.0


# --- Test Cases ---

print("--- Running Discontinuity Chain Tests ---")

# Test 1: Constant delay (t_k+1 = t_k + 1)
t0_ex1 = 0.0
tend_ex1 = 5.0
t_ex1 = [t0_ex1, tend_ex1]
h_guess_ex1 = 0.5  # Root is 1 unit away, this guess works
discs_ex1 = find_discontinuity_chain(alpha_const_delay, t_ex1, h_guess_ex1)
print(f"\nTest 1 (alpha(t)=t-1, t0={t0_ex1}, tf={tend_ex1}):")
print(f"  Found: {discs_ex1}")
print(f"  Expected: [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]")

# Test 2: Linear delay (t_k+1 = 2 * t_k)
t0_ex2 = 1.0
tend_ex2 = 10.0
t_ex2 = [t0_ex2, tend_ex2]
h_guess_ex2 = 0.5  # A small guess to find the next root
discs_ex2 = find_discontinuity_chain(alpha_linear_delay, t_ex2, h_guess_ex2)
print(f"\nTest 2 (alpha(t)=0.5*t, t0={t0_ex2}, tf={tend_ex2}):")
print(f"  Found: {discs_ex2}")
print(f"  Expected: [1.0, 2.0, 4.0, 8.0]")

# Test 3: Non-linear delay (t_k+1 = 2 * sqrt(t_k))
t0_ex3 = 0.5  # Needs t0 > 0
tend_ex3 = 6.0
t_ex3 = [t0_ex3, tend_ex3]
h_guess_ex3 = 0.1  # Guess needs to be carefully chosen for non-linear alpha
discs_ex3 = find_discontinuity_chain(alpha_nonlinear_delay, t_ex3, h_guess_ex3)
# Expected values: 0.5, 2*sqrt(0.5) approx 1.414, 2*sqrt(1.414) approx 2.378, 2*sqrt(2.378) approx 3.084, 2*sqrt(3.084) approx 3.512, 2*sqrt(3.512) approx 3.748...
print(f"\nTest 3 (alpha(t)=t^2/4, t0={t0_ex3}, tf={tend_ex3}):")
print(f"  Found: {discs_ex3}")
print(f"  Expected: [0.5, 1.4142..., 2.3784..., 3.0844..., 3.5124..., 3.7483...]")


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
t_ex4 = [t0_ex4, tend_ex4]
h_guess_ex4 = 0.1  # Small guess for robustness
discs_ex4 = find_discontinuity_chain(alpha_vector_mixed, t_ex4, h_guess_ex4)
print(f"\nTest 4 (alpha(t)=[t-1, 0.5t], t0={t0_ex4}, tf={tend_ex4}):")
print(f"  Found: {discs_ex4}")
print(f"  Expected: [0.5, 1.0, 2.0, 3.0, 4.0, 5.0]")

# Test 5: Square root and linear vector delays
t0_ex5 = 2.0
tend_ex5 = 30.0
t_ex5 = [t0_ex5, tend_ex5]
h_guess_ex5 = 0.1  # Small guess for robustness
discs_ex5 = find_discontinuity_chain(alpha_vector_sqrt_linear, t_ex5, h_guess_ex5)
print(f"\nTest 5 (alpha(t)=[sqrt(t), t/2], t0={t0_ex5}, tf={tend_ex5}):")
print(f"  Found: {discs_ex5}")
print(f"  Expected: [2.0, 4.0, 8.0, 16.0]")

print("\n--- Tests Finished ---")


print("\n--- Tests Finished ---")
