from methods import *
import numpy as np

def line():
    print("="*15)

# Define the system of equations
A = np.array([
    [3, -5, 47, 20],
    [11, 16, 17, 10],
    [56, 22, 11, -18],
    [17, 66, -12, 7]
])
b = np.array([18, 26, 34, 82])

def main():
    line()
    print("Solving the system using various methods:")
    line()

    # Solve using Cramer's method
    try:
        x_cramer = cramer_method(A, b)
        print("Solution using Cramer's Method:", x_cramer)
    except ValueError as e:
        print("Cramer's Method Error:", e)

    line()

    # Solve using Gaussian Elimination
    try:
        x_gaussian = gaussian_elimination(A.copy(), b.copy())
        print("Solution using Gaussian Elimination:", x_gaussian)
    except ValueError as e:
        print("Gaussian Elimination Error:", e)

    line()

    # Solve using Jacobi Iterative Method
    try:
        x_jacobi = jacobi_iteration(A, b)
        print("Solution using Jacobi Iteration:", x_jacobi)
    except Exception as e:
        print("Jacobi Iteration Error:", e)

    line()

    # Solve using Gauss-Seidel Method
    try:
        augmented_matrix = np.hstack((A, b.reshape(-1, 1)))
        x_gauss_seidel = gauss_seidel(augmented_matrix, len(b))
        print("Solution using Gauss-Seidel Method:", x_gauss_seidel)
    except Exception as e:
        print("Gauss-Seidel Method Error:", e)

if __name__ == "__main__":
    main()
