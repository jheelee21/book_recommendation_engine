import pandas as pd
import os
from pathlib import Path
import pickle

BOOK_DATA_FILE_PATH = "data/goodreads_data.csv"
BOOK_DATA_CACHE_FILE_PATH = "cache/book_data_cache.pkl"

def load_data(file_path: str) -> pd.DataFrame:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    
    return pd.read_csv(file_path)

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    # ,Book,Author,Description,Genres,Avg_Rating,Num_Ratings,URL

    if df.empty:
        raise ValueError("The DataFrame is empty.")

    df = df.dropna(subset=["Book", "Description"])

    df["Genres"] = df["Genres"].apply(lambda x: eval(x) if isinstance(x, str) else x)
    
    return df

if __name__ == "__main__":
    data = load_data(BOOK_DATA_FILE_PATH)
    processed_data = preprocess_data(data)
    print(processed_data.head())
    processed_data.to_pickle(BOOK_DATA_CACHE_FILE_PATH)