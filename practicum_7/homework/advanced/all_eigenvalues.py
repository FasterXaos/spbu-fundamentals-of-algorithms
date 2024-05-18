from collections import defaultdict
from dataclasses import dataclass
import os
import yaml
import time


import numpy as np
import scipy.io
import scipy.linalg

from src.common import NDArrayFloat
from src.linalg import get_numpy_eigenvalues


@dataclass
class Performance:
    time: float = 0.0
    relative_error: float = 0.0


def householder_tridiagonalization(A: NDArrayFloat) -> NDArrayFloat:
    n = A.shape[0]
    A = A.copy()
    
    for k in range(n - 2):
        x = A[k+1:, k]
        e1 = np.zeros_like(x)
        e1[0] = np.linalg.norm(x) * np.sign(x[0])
        v = x + e1
        v_norm = np.linalg.norm(v)
        
        if v_norm != 0:
            v = v / v_norm
        
            Hk = np.eye(n)
            Hk[k+1:, k+1:] -= 2.0 * np.outer(v, v)
        
            A = Hk @ A @ Hk.T
        print(k, n, sep=" ")
    
    return A


def qr(A):
    n = A.shape[0]
    Q = np.zeros_like(A)
    R = np.zeros_like(A)
    for i in range(n):
        v = A[:, i]

        for j in range(i):
            R[j, i] = np.dot(Q[:, j], A[:, i])
            v -= R[j, i] * Q[:, j]
            
        R[i, i] = np.linalg.norm(v)
        Q[:, i] = v / R[i, i]
        #print(i, n, sep=" ")
    #print(Q, R, Q@R, sep='\n')
    return Q, R


def qr_tridiagonal(A):
    n = A.shape[0]
    Q = np.eye(n)
    R = A.copy()

    for i in range(n - 1):
        x = R[i:i+2, i]
        norm_x = np.linalg.norm(x)
        if norm_x != 0:
            c, s = x[0] / norm_x, -x[1] / norm_x
        else:
            c, s = 1.0, 0.0
        
        for j in range(i, n):
            tau1, tau2 = R[i, j], R[i + 1, j]
            R[i, j] = c * tau1 - s * tau2
            R[i + 1, j] = s * tau1 + c * tau2
        
        for j in range(n):
            tau1, tau2 = Q[j, i], Q[j, i + 1]
            Q[j, i] = c * tau1 - s * tau2
            Q[j, i + 1] = s * tau1 + c * tau2

        #print(i, n, sep=" ")

    return Q, R


def get_all_eigenvalues(A: NDArrayFloat, n_iters: int = 5) -> NDArrayFloat:
    #A_k = A.copy()
    A_k = householder_tridiagonalization(A)

    for _ in range(n_iters):
        #Q, R = qr(A_k)
        Q, R = qr_tridiagonal(A_k)
        A_k = R @ Q

    eigenvalues = []
    n = A_k.shape[0]
    i = 0
    while i < n:
        if i < n - 1 and np.abs(A_k[i, i+1]) > 1e-10:
            a = A_k[i, i]
            b = A_k[i, i+1]
            c = A_k[i+1, i]
            d = A_k[i+1, i+1]
            discr = (a + d)**2 - 4*(a*d - b*c)
            eigenvalue1 = ((a + d) + np.lib.scimath.sqrt(discr)) / 2
            eigenvalue2 = ((a + d) - np.lib.scimath.sqrt(discr)) / 2
            eigenvalues.append(eigenvalue1)
            eigenvalues.append(eigenvalue2)
            i += 2
        else:
            eigenvalues.append(A_k[i, i])
            i += 1

    return np.array(eigenvalues)



def run_test_cases(
    path_to_homework: str, path_to_matrices: str
) -> dict[str, Performance]:
    matrix_filenames = []
    performance_by_matrix = defaultdict(Performance)
    with open(os.path.join(path_to_homework, "matrices.yaml"), "r") as f:
        matrix_filenames = yaml.safe_load(f)
    for i, matrix_filename in enumerate(matrix_filenames):
        print(f"Processing matrix {i+1} out of {len(matrix_filenames)}")
        A = scipy.io.mmread(os.path.join(path_to_matrices, matrix_filename)).todense().A
        perf = performance_by_matrix[matrix_filename]
        t1 = time.time()
        eigvals = get_all_eigenvalues(A)
        t2 = time.time()
        perf.time += t2 - t1
        eigvals_exact = get_numpy_eigenvalues(A)
        eigvals_exact = np.sort(eigvals_exact)
        eigvals_sorted = np.sort(eigvals)
        for i, (val_sorted, val_exact) in enumerate(zip(eigvals_sorted, eigvals_exact)):
            print(f"{i+1}. Sorted: {val_sorted}, Exact: {val_exact}")
        perf.relative_error = np.median(
            np.abs(eigvals_exact - eigvals_sorted) / np.abs(eigvals_exact)
        )
    return performance_by_matrix


if __name__ == "__main__":
    path_to_homework = os.path.join("practicum_7", "homework", "advanced")
    path_to_matrices = os.path.join("practicum_6", "homework", "advanced", "matrices")
    performance_by_matrix = run_test_cases(
        path_to_homework=path_to_homework,
        path_to_matrices=path_to_matrices,
    )

    print("\nResult summary:")
    for filename, perf in performance_by_matrix.items():
        print(
            f"Matrix: {filename}. "
            f"Average time: {perf.time:.2e} seconds. "
            f"Relative error: {perf.relative_error:.2e}"
        )
