import queue
from typing import Any

import networkx as nx

from src.plotting import plot_graph


def visit(node: Any):
    print(f"Wow, it is {node} right here!")


def dfs_recursive(G: nx.Graph, node: Any, visited: dict[Any]) -> None:
    visited[node] = True
    visit(node)
        
    for neighbor in G.neighbors(node):
        if not visited[neighbor]:
            dfs_recursive(G, neighbor, visited)


def dfs_iterative(G: nx.Graph, node: Any) -> None:
    visited = {node: False for node in G.nodes()}  # Initialize visited dictionary
    stack = [node]  # Initialize stack with starting node
    while stack:
        current_node = stack.pop()  # Pop the top node from the stack
        if not visited[current_node]:
            visited[current_node] = True
            visit(current_node)
            # Push unvisited neighbors onto the stack
            for neighbor in G.neighbors(current_node):
                if not visited[neighbor]:
                    stack.append(neighbor)


def dfs_recursive_postorder(G: nx.DiGraph, node: Any, visited: dict[Any]) -> None:
    if not visited[node]:
        visited[node] = True
        for neighbor in G.neighbors(node):
            if not visited[neighbor]:
                dfs_recursive_postorder(G, neighbor, visited)
        visit(node)


if __name__ == "__main__":
    G = nx.read_edgelist("practicum_2/graph_2.edgelist", create_using=nx.Graph)
    # plot_graph(G)

    # 1. Recursive DFS. Trivial to implement, but it does not scale on large graphs
    # In the debug mode, look at the call stack
    print("Recursive DFS")
    print("-" * 32)
    visited = {n: False for n in G}
    dfs_recursive(G, node="0", visited=visited)
    print()

    # 2. Iterative DFS. Makes use of LIFO/stack data structure, does scale on large graphs
    print("Iterative DFS")
    print("-" * 32)
    dfs_iterative(G, node="0")
    print()

    # 3. Postorder recursive DFS for topological sort
    # If a directed graph represent tasks to be done, the topological sort tells
    # us what the task order should be, i.e. scheduling
    # Postorder DFS outputs the reversed order!
    G = nx.read_edgelist("practicum_2/graph_2.edgelist", create_using=nx.DiGraph)
    plot_graph(G)
    print("Postorder iterative DFS")
    print("-" * 32)
    visited = {n: False for n in G}
    dfs_recursive_postorder(G, node="0", visited=visited)
