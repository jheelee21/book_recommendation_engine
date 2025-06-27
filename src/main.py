import pandas as pd

from src.utils.data_preprocessing import load_book_data
from src.embedding.embedding_model import OpenAIEmbeddingModel, BOOK_EMBEDDING_CACHE_FILE_PATH
from src.approx_nearest_neighbours.ann import ApproximateNearestNeighbours



def main():
    book_data = load_book_data()
    book_data = book_data.head(100)

    description_embedding_model = OpenAIEmbeddingModel(
        cache_file_path=BOOK_EMBEDDING_CACHE_FILE_PATH,
        dimensions=100,
    )

    embedding_df = pd.DataFrame(
        book_data["Description"].apply(description_embedding_model.get_or_create_embedding)
    )

    ## TODO: store df to pickle file, with book id and embedding
    ## TODO: fetching book info based on id
    ## TODO: experiment on weights