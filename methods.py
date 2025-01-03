import numpy as np


def cramer_method(A, b):
    """
    Solves a system of linear equations Ax = b using Cramer's Method.

    Parameters:
    A : np.array : Coefficient matrix (square matrix)
    b : np.array : Right-hand side vector

    Returns:
    x : np.array : Solution to the system of equations
    """

    n = A.shape[0]
    if A.shape[1] != n:
        raise ValueError("Matrix A must be square")

    # Determinant
    det_A = np.linalg.det(A)
    if det_A == 0:
        raise ValueError(f"The determinant of matrix A is 0. The system has no unique solution.")

    # Solution vector
    x = np.zeros(n)

    # Loop through each column
    for k in range(n):
        D = A.copy()
        D[:, k] = b  # Replace ka to kb
        det_D = np.linalg.det(D)  # Determinant modified
        x[k] = det_D / det_A # k by cramer

    return x


def gaussian_elimination(A, b):
    """
    Solves the linear system Ax = b using Gaussian elimination.

    Parameters:
    A : np.array : Coefficient matrix
    b : np.array : Right-hand side vector

    Returns:
    x : np.array : Solution vector
    """
    n = len(b)
    for k in range(n - 1):
        for i in range(k + 1, n):
            if A[k][k] == 0:
                raise ValueError("Division by zero detected!")
            factor = A[i][k] / A[k][k]  # Compute c_ik
            for j in range(k, n):
                A[i][j] = A[i][j] - factor * A[k][j]  # a_ij = a_ij - c_ik * a_kj
            b[i] = b[i] - factor * b[k]  # b_i = b_i - c_ik * b_k

    # Back Substitution
    x = np.zeros(n)
    x[n - 1] = b[n - 1] / A[n - 1][n - 1]  # Starting with x_n
    for i in range(n - 2, -1, -1):
        sum_ax = 0
        for j in range(i + 1, n):
            sum_ax += A[i][j] * x[j]
        x[i] = (b[i] - sum_ax) / A[i][i]  # x_i = (b_i - sum) / a_ii

    return x


def jacobi_iteration(a, b, n_iter=25, tol=1e-10):
    """
    Solves a system of linear equations Ax = b using Jacobi's Iterative Method.

    Parameters:
        a (2D list or numpy array): Coefficient matrix A (n x n).
        b (1D list or numpy array): Right-hand side vector b (n x 1).
        n_iter (int): Maximum number of iterations (default = 25).
        tol (float): Convergence tolerance for stopping (default = 1e-10).

    Returns:
        x (numpy array): Solution vector x.
    """
    n = len(b)
    x = np.zeros(n)
    x_new = np.zeros(n)

    for m in range(n_iter):
        for i in range(n):
            sigma = sum(a[i][j] * x[j] for j in range(n) if i != j)
            x_new[i] = (b[i] - sigma) / a[i][i]

        # Check convergence (stopping criterion)
        if np.linalg.norm(x_new - x, ord=np.inf) < tol:
            print(f"Converged in {m + 1} iterations.")
            return x_new

        x = x_new.copy()

    print("Maximum iterations reached without convergence.")
    return x_new


def gauss_seidel(a, b, n_iterations=25, tolerance=1e-4):
    """
    Solves a system of linear equations Ax = b using the Gauss-Seidel Method.

    Parameters:
        a (numpy.ndarray): Coefficient matrix (n x n)
        b (numpy.ndarray): Constant terms vector (n x 1)
        n_iterations (int): Maximum number of iterations
        tolerance (float): Convergence tolerance

    Returns:
        numpy.ndarray: Solution vector x
    """
    n = len(b)
    x = np.zeros(n)
    y = np.zeros(n)

    for itr in range(1, n_iterations + 1):
        for i in range(n):
            sum1 = sum(a[i][j] * x[j] for j in range(n) if j != i)
            x[i] = (b[i] - sum1) / a[i][i]

        # Check convergence (stopping criterion)
        if all(abs(x[k] - y[k]) <= tolerance for k in range(n)):
            print(f"Converged in {itr} iterations.")
            return x

        y = x.copy()

    print("Maximum iterations reached without convergence.")
    return x