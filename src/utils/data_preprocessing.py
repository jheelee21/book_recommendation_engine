import pandas as pd
import os
from pathlib import Path
import pickle

BOOK_DATA_FILE_PATH = "data/goodreads_data.csv"
BOOK_DATA_CACHE_FILE_PATH = "cache/book_data_cache.pkl"

def load_book_data() -> pd.DataFrame:
    if os.path.exists(BOOK_DATA_CACHE_FILE_PATH):
        with open(BOOK_DATA_CACHE_FILE_PATH, "rb") as cache_file:
            return pickle.load(cache_file)
        
    if not os.path.exists(BOOK_DATA_FILE_PATH):
        raise FileNotFoundError(f"The file {BOOK_DATA_FILE_PATH} does not exist.")
    
    df = pd.read_csv(BOOK_DATA_FILE_PATH)
    df = preprocess_data(df)
    df.to_pickle(BOOK_DATA_CACHE_FILE_PATH)
    return df

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    # ,Book,Author,Description,Genres,Avg_Rating,Num_Ratings,URL
    if df.empty:
        raise ValueError("The DataFrame is empty.")

    df = df.dropna(subset=["Book", "Description"])

    df["Genres"] = df["Genres"].apply(lambda x: eval(x) if isinstance(x, str) else x)
    
    return df

if __name__ == "__main__":
    data = load_book_data()
    print(len(data))