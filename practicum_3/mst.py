from typing import Any

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from src.plotting import plot_graph


def prim_mst(G: nx.Graph, start_node="0") -> set[tuple[Any, Any]]:
    mst_set = {start_node}  # set of nodes included into MST
    rest_set = set(G.nodes()) - mst_set  # set of nodes not yet included into MST
    mst_edges = set()  # set of edges constituting MST

    while rest_set:
        min_weight = float('inf')
        min_edge = None

        # Ищем ребро минимального веса, соединяющее вершину из MST с вершиной за его пределами
        for node in mst_set:
            for neighbor, weight in G[node].items():
                if neighbor in rest_set and weight['weight'] < min_weight:
                    min_weight = weight['weight']
                    min_edge = (node, neighbor)

        # Добавляем найденное ребро в MST
        mst_edges.add(min_edge)
        mst_set.add(min_edge[1])
        rest_set.remove(min_edge[1])

    return mst_edges


if __name__ == "__main__":
    G = nx.read_edgelist("practicum_3/graph_1.edgelist", create_using=nx.Graph)
    plot_graph(G)
    mst_edges = prim_mst(G, start_node="0")
    plot_graph(G, highlighted_edges=list(mst_edges))
