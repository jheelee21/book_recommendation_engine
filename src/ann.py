from utils.kd_tree import Node, build_tree, search
import pickle
import numpy as np


def save_tree(tree: Node, file_path: str) -> None:
    with open(file_path, "wb") as f:
        pickle.dump(tree, f)


def load_tree(file_path: str) -> Node:
    with open(file_path, "rb") as f:
        return pickle.load(f)


def get_k_approx_nearest_neighbors(
    tree: Node, query_vector: np.ndarray, k: int
) -> list[np.ndarray]:
    neighbors = search(tree, query_vector)
    return sorted(neighbors, key=lambda x: np.linalg.norm(x - query_vector))[:k]


def build_embedding_tree_from_cache(cache: dict, file_path: str, min_subset_size: int = 5) -> Node:
    vectors = list(cache.values())
    tree = build_tree(vectors, min_subset_size)
    save_tree(tree, file_path)
    return tree


if __name__ == "__main__":
    # Example usage
    vectors = [np.random.rand(3) for _ in range(100)]
    test_tree = build_tree(vectors)

    tree_cache_file_path = "utils/data/kd_tree_test.pkl"

    save_tree(test_tree, tree_cache_file_path)

    loaded_tree = load_tree(tree_cache_file_path)

    sample = np.random.rand(3)
    nearest_neighbors = get_k_approx_nearest_neighbors(loaded_tree, sample, k=5)

    print(f"neighbors for sample {sample} are \n {nearest_neighbors}")
