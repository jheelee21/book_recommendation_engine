from openai import OpenAI
import pandas as pd
import pickle
import os
from dotenv import load_dotenv
from pathlib import Path
import numpy as np

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_CACHE_FILE_PATH = "cache/data/book_embeddings_cache.pkl"

class OpenAIEmbeddingModel:
    def __init__(
        self,
        cache_file_path: str = EMBEDDING_CACHE_FILE_PATH,
        model: str = EMBEDDING_MODEL,
        dimensions: int = 100,
    ):
        self.client = self.config_openai()
        self.cache_file_path = cache_file_path
        self.embedding_cache = self.load_embedding_cache(cache_file_path)
        self.model = model
        self.dimensions = dimensions

    def config_openai(self) -> OpenAI:
        load_dotenv()
        openai_api_key = os.getenv("OPENAI_API_KEY")
        client = OpenAI(api_key=openai_api_key)
        return client

    def load_embedding_cache(self) -> dict:
        Path(os.path.dirname(self.cache_file_path)).mkdir(parents=True, exist_ok=True)

        if os.path.exists(self.cache_file_path):
            embedding_cache = pd.read_pickle(self.cache_file_path)
        else:
            embedding_cache = {}

        return embedding_cache

    def get_embedding_from_text(self, text: str) -> list[float]:
        text = text.replace("\n", " ")
        result = self.client.embeddings.create(
            input=text, model=self.model, dimensions=self.dimensions
        )
        print(f"Usage: {result.usage}")
        return result.data[0].embedding

    def get_or_create_embedding(self, text: str) -> list[float]:
        if (text, self.model) not in self.embedding_cache.keys():
            self.embedding_cache[(text, self.model)] = self.get_embedding_from_text(
                text
            )
            with open(self.embedding_cache_file_path, "wb") as embedding_cache_file:
                pickle.dump(self.embedding_cache, embedding_cache_file)
        return self.embedding_cache[(text, self.model)]


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


if __name__ == "__main__":
    # Example usage
    text1 = "Some ghosts don't seek peace—they demand the truth.\n\nWhen Meera returns to her hometown after the mysterious deaths of her sister Radhika and young nephew Rajiv, she expects grief. Instead, she finds secrets—buried deep in family shadows, whispered in dreams, and painted on the walls of a forgotten temple.\n\nWith the help of a haunted ex-cop, Meera uncovers an ancient cult, a bloodline marked for sacrifice, and a ritual that never ended. But the deeper they dig, the more the past fights back.\n\nPhantasm is a gripping supernatural thriller about love, legacy, and the darkness that waits when truth is left to rot. Some endings demand blood. Others demand the courage to face them."
    book_id1 = "27"

    text2 = "What would you do if you uncovered the secret of a hidden treasure? Would you chase the gold—or hunt down the war criminals who buried it?\n\nFrom the bestselling author of The Hunter series and numerous other novels, Jasveer Singh Dangi, comes a heart-pounding tale that will keep you on the edge of your seat.\n\nIn this fast-paced thriller, a cryptic message sparks a high-stakes cat and mouse chase that sends a diverse team of experts on a dangerous race to unearth both Nazi war criminals and a treasure lost for over 55 years in Canada.\n\nLed by a battle-hardened former Special Forces operative, a seasoned treasure hunter with a shadowy past, and a brilliant young idealist driven by a relentless thirst for justice, the team must track down elusive war criminals who have remained hidden in plain sight for decades. These criminals have evaded justice for far too long—and now it's time for them to face the consequences.\n\nOn this epic rollercoaster ride, the team uncovers shocking secrets and decodes long-buried, deliberately erased clues. As the fate of both the treasure and the criminals hangs in the balance, they race against time to bring justice to the shadows of the past.\n\nPrepare for a pulse-pounding adventure filled with relentless suspense, unexpected twists, and a high-stakes quest to uncover the hidden Nazi gold before it falls into the wrong hands."
    book_id2 = "26"

    text3 = "We live in confounding times that the author attempts to explain with incisive analyses, broad criticism and boundless humor. His compilation of essays course through our cultural, social, political and financial milieus. A broad range of topics are covered from student debt to gun safety, from bitcoin to the demise of the internal combustion engine, and from capitalism to empty calories; and much more. His irreverence is a worthy match for pervasive absurdity."
    book_id3 = "25"

    genre1 = "Science Fiction"
    genre2 = "Romance"
    genre3 = "Non-fiction"
    genre4 = "Thriller"

    book_embedding_model = OpenAIEmbeddingModel()
    embeddings = [
        book_embedding_model.get_or_create_embedding(text)
        for text in [text1, text2, text3]
    ]

    genre_embedding_model = OpenAIEmbeddingModel(
        cache_file_path="utils/data/genre_embeddings_cache.pkl", dimensions=8
    )
    genre_embeddings = [
        genre_embedding_model.get_or_create_embedding(genre)
        for genre in [genre1, genre2, genre3, genre4]
    ]

    genre_df = pd.DataFrame({"id": range(1, 5), "embedding": genre_embeddings})
    print(naive_get_k_similar_items(1, genre_embeddings[0], genre_df, k=3))

    df = pd.DataFrame({"id": [book_id1, book_id2, book_id3], "embedding": embeddings})
    print(naive_get_k_similar_items(book_id1, embeddings[0], df, k=2))

    print("Cosine Similarities:")
    print(cosine_similarity(embeddings[0], embeddings[1]))
    print(cosine_similarity(embeddings[0], embeddings[2]))
    print(cosine_similarity(embeddings[1], embeddings[2]))
