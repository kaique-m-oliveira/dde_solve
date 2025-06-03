import numpy as np

# The matrix A provided by you
A_matrix_vals = [
    [0.1968154772236604, -0.0655354258501984, 0.0237709743482202],
    [0.3944243147390873, 0.2920734116652284, -0.0415487521259979],
    [0.3764030627004673, 0.5124858261884216, 0.1111111111111111],
]

# Convert the list of lists to a NumPy array
A_matrix = np.array(A_matrix_vals)

# --- Check if the matrix is invertible (optional, but good practice) ---
# A matrix is invertible if its determinant is non-zero.
determinant_A = np.linalg.det(A_matrix)

print("Original Matrix A:")
print(A_matrix)
print(f"\nDeterminant of A: {determinant_A}")

if determinant_A == 0:
    print("\nThe matrix A is singular and cannot be inverted.")
else:
    # --- Invert the matrix A ---
    A_inv = np.linalg.inv(A_matrix)

    print("\nInverse of Matrix A (A_inv):")
    print(A_inv)

    # --- Verify the result (optional): A @ A_inv should be close to the identity matrix ---
    # In Python 3.5+ you can use the @ operator for matrix multiplication
    identity_check = A_matrix @ A_inv
    print("\nVerification (A @ A_inv, should be close to Identity Matrix):")
    print(identity_check)

    # If you want the inverse matrix as a list of lists:
    A_inv_list = A_inv.tolist()
    print("\nInverse of Matrix A (as list of lists):")
    for row in A_inv_list:
        print(f"    [{', '.join(f'{x:.16f}' for x in row)}],")
