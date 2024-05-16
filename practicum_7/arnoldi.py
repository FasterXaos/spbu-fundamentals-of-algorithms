import numpy as np
import matplotlib.pyplot as plt

from src.common import NDArrayFloat


def get_arnoldi_vectors(A: NDArrayFloat, n_iters: int) -> NDArrayFloat:
    n = A.shape[0]
    m = min(n, n_iters + 1)

    arnoldi_vectors = np.zeros((n, m))

    v = np.random.rand(n)
    v /= np.linalg.norm(v)
    arnoldi_vectors[:, 0] = v

    for j in range(1, m):
        w = A @ arnoldi_vectors[:, j - 1]  # Умножаем матрицу A на последний вектор Арнольди
        for i in range(j):  # Ортогонализуем полученный вектор относительно предыдущих векторов
            h_ij = np.dot(w, arnoldi_vectors[:, i])
            w -= h_ij * arnoldi_vectors[:, i]
        h_jj = np.linalg.norm(w)
        if h_jj == 0:  # Обработка исключения, чтобы избежать деления на ноль
            break
        arnoldi_vectors[:, j] = w / h_jj

    return arnoldi_vectors


def check_arnoldi_vectors(arnoldi_vectors, A):
    dot_products = np.dot(arnoldi_vectors.T, arnoldi_vectors)
    orthogonality_error = np.max(np.abs(dot_products - np.eye(dot_products.shape[0])))
    print("Ортогональность столбцов: максимальная ошибка =", orthogonality_error)

    norms = np.linalg.norm(arnoldi_vectors, axis=0)
    norm_error = np.max(np.abs(norms - 1))
    print("Норма каждого вектора: максимальная ошибка =", norm_error)

    V = arnoldi_vectors
    H = V.T @ A @ V
    AV_minus_VH = A @ V - V @ H
    av_minus_vh_error = np.max(np.abs(AV_minus_VH))
    print("Проверка AV - VH: максимальная ошибка =", av_minus_vh_error)


def find_eigenvalues_with_arnoldi(A, n_iters):
    arnoldi_vectors = get_arnoldi_vectors(A, n_iters)
    H = arnoldi_vectors.T @ A @ arnoldi_vectors  # Верхнетреугольная матрица Хессенберга
    eigenvalues = get_eigenvalues_via_qr(H, n_iters)  # Находим собственные значения матрицы H
    return eigenvalues


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
    #print(Q, R, Q@R, sep='\n')
    return Q, R


def get_eigenvalues_via_qr(A: NDArrayFloat, n_iters: int) -> NDArrayFloat:
    A_k = A.copy()
    for _ in range(n_iters):
        Q, R = qr(A_k)
        A_k = R @ Q
    return  np.diag(A_k) 


if __name__ == "__main__":
    A = np.array(
        [
            [4.0, 1.0, -1.0, 2.0],
            [1.0, 4.0, 1.0, -1.0],
            [-1.0, 1.0, 4.0, 1.0],
            [2.0, -1.0, 1.0, 1.0],
        ]
    )

    Q = get_arnoldi_vectors(A, n_iters=3)
    check_arnoldi_vectors(Q, A)

    eigenvalues = find_eigenvalues_with_arnoldi(A, n_iters=20)
    print("Собственные значения матрицы A:", eigenvalues)