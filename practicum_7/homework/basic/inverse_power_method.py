import numpy as np
import matplotlib.pyplot as plt

from src.common import NDArrayFloat


def inverse_power_method(A: NDArrayFloat, mu: float, n_iters: int) -> float:
    n = A.shape[0]
    x = np.random.rand(n)
    x /= np.linalg.norm(x)

    # Находим обратную матрицу (A - mu*I)^(-1)
    inverse_matrix = np.linalg.inv(A - mu * np.eye(n))

    for _ in range(n_iters):
        x = inverse_matrix @ x  # Умножаем обратную матрицу на вектор
        x /= np.linalg.norm(x)  # Нормируем вектор

    # Вычисляем собственное значение, используя отношение Релея
    eigenvalue = np.dot(x, A @ x)

    return eigenvalue

if __name__ == "__main__":
    A = np.array(
        [
            [4.0, 1.0, -1.0, 2.0],
            [1.0, 4.0, 1.0, -1.0],
            [-1.0, 1.0, 4.0, 1.0],
            [2.0, -1.0, 1.0, 1.0],
        ]
    )
    mu = 1.3  # Ищем собственный вектор будет максимально близrbq к mu
    eigvalue = inverse_power_method(A, mu, n_iters=50)
    print("Собственные значения:", eigvalue)