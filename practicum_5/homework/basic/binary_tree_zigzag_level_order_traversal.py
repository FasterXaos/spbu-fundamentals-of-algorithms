from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import os

import yaml


@dataclass
class Node:
    key: Any
    data: Any = None
    left: Node = None
    right: Node = None


class BinaryTree:
    def __init__(self) -> None:
        self.root: Node = None

    def empty(self) -> bool:
        return self.root is None

    def zigzag_level_order_traversal(self) -> list[Any]:
        if self.empty():
            return []

        result = []
        level = 0
        queue = [self.root]

        while queue:
            level_values = []
            next_level = []

            while queue:
                node = queue.pop(0)
                level_values.append(node.key)
                if node.left:
                    next_level.append(node.left)
                if node.right:
                    next_level.append(node.right)
            if level % 2 == 1:
                level_values.reverse()

            result.append(level_values)
            queue = next_level
            level += 1
            
        return result


def build_tree(list_view: list[Any]) -> BinaryTree:
    bt = BinaryTree()

    if not list_view:
        return bt

    root = Node(list_view[0])
    bt.root = root
    queue = [root]
    i = 1

    while queue and i < len(list_view):
        current = queue.pop(0)
        if list_view[i] is not None:
            left_node = Node(list_view[i])
            current.left = left_node
            queue.append(left_node)
        i += 1
        if i < len(list_view) and list_view[i] is not None:
            right_node = Node(list_view[i])
            current.right = right_node
            queue.append(right_node)
        i += 1

    return bt


if __name__ == "__main__":
    # Let's solve Binary Tree Zigzag Level Order Traversal problem from leetcode.com:
    # https://leetcode.com/problems/binary-tree-zigzag-level-order-traversal/
    # First, implement build_tree() to read a tree from a list format to our class
    # Second, implement BinaryTree.zigzag_traversal() returning the list required by the task
    # Avoid recursive traversal!

    with open(
        os.path.join(
            "practicum_5",
            "homework",
            "basic",
            "binary_tree_zigzag_level_order_traversal_cases.yaml",
        ),
        "r",
    ) as f:
        cases = yaml.safe_load(f)

    for i, c in enumerate(cases):
        bt = build_tree(c["input"])
        zz_traversal = bt.zigzag_level_order_traversal()
        print(f"Case #{i + 1}: {zz_traversal == c['output']}")
