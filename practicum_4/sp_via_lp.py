import networkx as nx
import numpy as np

from src.plotting import plot_graph
from scipy.optimize import linprog


def solve_via_lp(G, s_node, t_node):
    num_nodes = len(G.nodes)
    num_edges = len(G.edges)
    
    # Создаем массив весов рёбер
    w = np.array([G[u][v]['weight'] for u, v in G.edges])
    
    # Создаем матрицу смежности
    A = np.zeros((num_nodes, num_edges))
    for i, (u, v) in enumerate(G.edges):
        A[int(u), i] = 1
        A[int(v), i] = -1
    
    # Создаем вектор правой части для ограничений равенства
    b_eq = np.zeros(num_nodes)
    b_eq[int(s_node)] = 1
    b_eq[int(t_node)] = -1
    
    # Создаем ограничения на неотрицательность переменных
    bounds = [(0, None)] * num_edges
    
    # Решаем линейную программу
    res = linprog(w, A_eq=A, b_eq=b_eq, bounds=bounds)
    
    # Извлекаем индексы рёбер, которые входят в кратчайший путь
    shortest_path_indices = np.where(res.x > 0.5)[0]
    
    # Формируем список рёбер кратчайшего пути
    shortest_path_edges = [(list(G.edges)[i][0], list(G.edges)[i][1]) for i in shortest_path_indices]
    
    return shortest_path_edges


if __name__ == "__main__":
    G = nx.read_edgelist("practicum_4/graph.edgelist", create_using=nx.Graph)
    #plot_graph(G)

    s_node = "0"
    t_node = "3"
    shortest_path_edges = solve_via_lp(G, s_node=s_node, t_node=t_node)
    plot_graph(G, highlighted_edges=shortest_path_edges)
