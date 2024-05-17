from operator import itemgetter
from queue import PriorityQueue
from typing import Any

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from src.plotting import plot_graph


def dijkstra_sp_without_priority_queue(G: nx.Graph, source_node="0") -> dict[Any, list[Any]]:
    unvisited_set = set(G.nodes())
    shortest_paths = {node: [source_node] if node == source_node else [] for node in G.nodes()}
    distances = {node: float('inf') if node != source_node else 0 for node in G.nodes()}

    while unvisited_set:
        min_distance = float('inf')
        min_node = None

        # Находим вершину с минимальным расстоянием из непосещенных
        for node in unvisited_set:
            if distances[node] < min_distance:
                min_distance = distances[node]
                min_node = node
        unvisited_set.remove(min_node)

        # Обновляем расстояния до соседей выбранной вершины
        for neighbor, edge_data in G[min_node].items():
            distance_to_neighbor = distances[min_node] + edge_data['weight']
            if distance_to_neighbor < distances[neighbor]:
                distances[neighbor] = distance_to_neighbor
                shortest_paths[neighbor] = shortest_paths[min_node] + [neighbor]

    return shortest_paths

def dijkstra_sp_with_priority_queue(G: nx.Graph, source_node="0") -> dict[Any, list[Any]]:
    unvisited_set = set(G.nodes())
    shortest_paths = {node: [source_node] if node == source_node else [] for node in G.nodes()}
    distances = {node: float('inf') if node != source_node else 0 for node in G.nodes()}
    
    pq = PriorityQueue()
    pq.put((0, source_node))

    while not pq.empty():
        min_distance, min_node = pq.get()
        if min_node not in unvisited_set:
            continue
        unvisited_set.remove(min_node)

        for neighbor, edge_data in G[min_node].items():
            distance_to_neighbor = distances[min_node] + edge_data['weight']
            if distance_to_neighbor < distances[neighbor]:
                distances[neighbor] = distance_to_neighbor
                shortest_paths[neighbor] = shortest_paths[min_node] + [neighbor]
                pq.put((distance_to_neighbor, neighbor))

    return shortest_paths

if __name__ == "__main__":
    G = nx.read_edgelist("practicum_3/graph_1.edgelist", create_using=nx.Graph)
    # plot_graph(G)
    shortest_paths = dijkstra_sp_with_priority_queue(G, source_node="0")
    test_node = "4"
    shortest_path_edges = [
        (shortest_paths[test_node][i], shortest_paths[test_node][i + 1])
        for i in range(len(shortest_paths[test_node]) - 1)
    ]
    plot_graph(G, highlighted_edges=shortest_path_edges)
