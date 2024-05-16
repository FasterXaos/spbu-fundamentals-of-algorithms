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
        print(i, n, sep=" ")
    #print(Q, R, Q@R, sep='\n')
    return Q, R


def get_all_eigenvalues(A: NDArrayFloat, n_iters: int = 3) -> NDArrayFloat:
    A_k = A.copy()
    eigenvalues = []
    n = A.shape[0]

    for m in range(n_iters):
        print(f"qr: {m+1}")
        Q, R = qr(A_k)
        A_k = R @ Q

    i = 0
    while i < n:
        if i < n - 1 and np.abs(A_k[i, i+1]) > 1e-10:
            # Блок 2x2 с комплексным собственным значением
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
            # Одиночное собственное значение
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
        #for i, (val_sorted, val_exact) in enumerate(zip(eigvals_sorted, eigvals_exact)):
        #    print(f"{i+1}. Sorted: {val_sorted}, Exact: {val_exact}")
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
