import random
import numpy as np


class Node:
    def __init__(
        self,
        hyperplane: np.ndarray | None = None,
        constant: float | None = None,
        values: list[np.ndarray] | None = None,
    ) -> None:
        self.hyperplane = hyperplane
        self.constant = constant
        self.values = values
        self.left = None
        self.right = None

    def __repr__(self):
        return f"Node(hyperplane={self.hyperplane}, constant={self.constant}, num_values={len(self.values) if self.values else 0})"


def hyperplane_equation(
    v1: np.ndarray, v2: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    normal_vector = v2 - v1
    midpoint = (v1 + v2) / 2
    const_term = np.dot(normal_vector, midpoint)
    return normal_vector, const_term


def is_vector_on_right_side(
    normal_vector: np.ndarray, constant: float, vector: np.ndarray
) -> bool:
    result = np.dot(normal_vector, vector)
    return result < constant


def build_tree(vectors: list[np.ndarray], min_subset_size: int = 5) -> Node:
    if len(vectors) <= min_subset_size or len(vectors) < 2:
        return Node(values=vectors)

    idx1 = idx2 = 0

    while idx1 == idx2:
        idx1, idx2 = random.sample(range(len(vectors)), 2)

    v1 = vectors[idx1]
    v2 = vectors[idx2]

    hyperplane, constant = hyperplane_equation(v1, v2)

    left_nodes = []
    right_nodes = []

    for vector in vectors:
        if is_vector_on_right_side(hyperplane, constant, vector):
            right_nodes.append(vector)
        else:
            left_nodes.append(vector)

    if not left_nodes or not right_nodes:  # no meaningful split
        return Node(values=vectors)

    current_node = Node(hyperplane=hyperplane, constant=constant, values=vectors)

    if len(left_nodes) > min_subset_size:
        current_node.left = build_tree(left_nodes, min_subset_size)
    else:
        current_node.left = Node(values=left_nodes)

    if len(right_nodes) > min_subset_size:
        current_node.right = build_tree(right_nodes, min_subset_size)
    else:
        current_node.right = Node(values=right_nodes)

    return current_node


def search(
    tree: Node, query_vector: np.ndarray, min_subset_size: int = 5
) -> list[np.ndarray]:
    while (
        len(tree.values) > min_subset_size
        and tree.hyperplane is not None
        and tree.constant is not None
    ):
        if is_vector_on_right_side(tree.hyperplane, tree.constant, query_vector):
            tree = tree.right
        else:
            tree = tree.left
    return tree.values


if __name__ == "__main__":
    # Example usage
    vectors = [np.random.rand(3) for _ in range(100)]
    test_tree = build_tree(vectors)
    sample = np.random.rand(3)
    nearest_neighbors = search(test_tree, sample)
    print(f"neighbors for sample {sample} are \n {nearest_neighbors}")
