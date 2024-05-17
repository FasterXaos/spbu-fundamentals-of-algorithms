import numpy as np
import matplotlib.pyplot as plt

from src.common import NDArrayFloat
from src.linalg import get_numpy_eigenvalues


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


def get_eigenvalues_via_qr(A: NDArrayFloat, n_iters: int = 100) -> NDArrayFloat:
    A_k = A.copy()
    eigenvalues = []
    n = A.shape[0]

    for _ in range(n_iters):
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


def householder_tridiagonalization(A: NDArrayFloat) -> NDArrayFloat:
    n = A.shape[0]
    A = A.copy()
    
    for k in range(n - 2):
        x = A[k+1:, k]
        e1 = np.zeros_like(x)
        e1[0] = np.linalg.norm(x) * np.sign(x[0])
        v = x + e1
        v = v / np.linalg.norm(v)
        
        Hk = np.eye(n)
        Hk[k+1:, k+1:] -= 2.0 * np.outer(v, v)
        
        A = Hk @ A @ Hk.T
    
    return A

def sign(x):
    return 1 if x > 0 else -1


if __name__ == "__main__":
    #A = np.random.rand(4, 4)
    A = np.array([[4, 1, -2, 2],
              [1, 2, 0, 1],
              [-2, 0, 3, -2],
              [2, 1, -2, -1]], dtype=float)
    
    eigvals = get_eigenvalues_via_qr(A, n_iters=10000)
    print(np.sort(eigvals))
    eigvals = get_numpy_eigenvalues(A)
    print (np.sort(eigvals))

    A_tri = householder_tridiagonalization(A)
    print(A_tri)
    eigvals_tri = get_eigenvalues_via_qr(A_tri, n_iters=20)
