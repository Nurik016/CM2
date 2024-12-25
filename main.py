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


# cramer_method
line()
try:
    x_cramer = cramer_method(A, b)
    print(f"Result of cramer_method: {x_cramer}")
except Exception as e:
    print(f"Error: {e}")


# gaussian_elimination
line()
try:
    x_gaussian = gaussian_elimination(A, b)
    print(f"Result of gaussian_elimination: {x_gaussian}")
except Exception as e:
    print(f"Error: {e}")

# jacobi_iteration
line()
try:
    x_jacobi = jacobi_iteration(A, b)
    print(f"Result of jacobi_iteration: {x_jacobi}")
except Exception as e:
    print(f"Error: {e}")

# gauss_seidel
line()
try:
    x_gauss = gauss_seidel(A, b)
    print(f"Result of gauss_seidel: {x_gauss}")
except Exception as e:
    print(f"Error: {e}")

