import numpy as np


def lu_factor(A):
    """LU decomposition with partial pivoting.
    Returns (L, U), P such that P@A = L@U
    """
    U = A.copy().astype(float)
    m = U.shape[0]
    L = np.eye(m)
    P = np.eye(m)

    for k in range(m - 1):
        # find pivot row
        idx = k + np.argmax(np.abs(U[k:m, k]))

        if U[idx, k] == 0:
            raise ValueError("Matrix is singular.")

        # swap rows in U
        U[[k, idx], k:m] = U[[idx, k], k:m]

        # swap rows in L (only the first k columns)
        if k > 0:
            L[[k, idx], :k] = L[[idx, k], :k]

        # swap rows in P
        P[[k, idx], :] = P[[idx, k], :]

        # elimination
        for j in range(k + 1, m):
            L[j, k] = U[j, k] / U[k, k]
            U[j, k:m] -= L[j, k] * U[k, k:m]

    return (L, U), P
