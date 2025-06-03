import math

# Calculate the square root of 6 once
sqrt6 = math.sqrt(6)

# --- c vector ---
c1 = (4 - sqrt6) / 10
c2 = (4 + sqrt6) / 10
c3 = 1.0

# --- A matrix ---
# Row 1
a11 = (88 - 7 * sqrt6) / 360
a12 = (296 - 169 * sqrt6) / 1800
a13 = (-2 + 3 * sqrt6) / 225

# Row 2
a21 = (296 + 169 * sqrt6) / 1800
a22 = (88 + 7 * sqrt6) / 360
a23 = (-2 - 3 * sqrt6) / 225

# Row 3
a31 = (16 - sqrt6) / 36
a32 = (16 + sqrt6) / 36
a33 = 1 / 9

# --- b vector ---
# (For Radau IIA, b_i = a_si, so these are the same as the last row of A)
b1 = (16 - sqrt6) / 36
b2 = (16 + sqrt6) / 36
b3 = 1 / 9

# --- Print the results in a tableau-like format ---
print("Radau IIA Coefficients (s=3, p=5) - Floating Point Approximations")
print("-" * 70)

print(f"c1 = {c1:.16f}")
print(f"c2 = {c2:.16f}")
print(f"c3 = {c3:.16f}")
print("-" * 70)

print("A matrix:")
print(f"a11 = {a11:.16f}, a12 = {a12:.16f}, a13 = {a13:.16f}")
print(f"a21 = {a21:.16f}, a22 = {a22:.16f}, a23 = {a23:.16f}")
print(f"a31 = {a31:.16f}, a32 = {a32:.16f}, a33 = {a33:.16f}")
print("-" * 70)

print("b vector (weights):")
print(f"b1 = {b1:.16f}, b2 = {b2:.16f}, b3 = {b3:.16f}")
print("-" * 70)

# For easier copy-pasting into code, here they are as lists/arrays
print("\nPython list format:")
print("c = [")
print(f"    {c1:.16f},")
print(f"    {c2:.16f},")
print(f"    {c3:.16f}")
print("]")

print("\nA = [")
print(f"    [{a11:.16f}, {a12:.16f}, {a13:.16f}],")
print(f"    [{a21:.16f}, {a22:.16f}, {a23:.16f}],")
print(f"    [{a31:.16f}, {a32:.16f}, {a33:.16f}]")
print("]")

print("\nb = [")
print(f"    {b1:.16f}, {b2:.16f}, {b3:.16f}")
print("]")
