import pandas as pd
import numpy as np

def cosine_similarity(a: list[float], b: list[float]) -> float:
    a = np.array(a)
    b = np.array(b)

    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot_product / (norm_a * norm_b)


def naive_get_k_similar_items(
    id: str, embedding: list[float], df: pd.DataFrame, k: int = 5
) -> list[str]:
    if id in df["id"].values:
        df = df[df["id"] != id].copy()

    df["similarity"] = df["embedding"].apply(lambda x: cosine_similarity(embedding, x))

    df = df.sort_values(by="similarity", ascending=False)
    related_books = df.head(k)["id"].tolist()
    print(df["similarity"])
    return related_books