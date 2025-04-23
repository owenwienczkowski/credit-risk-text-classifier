from src.load_data import download_raw_data
from src.preprocess import clean

if __name__ == "__main__":
        raw_data = download_raw_data()
        clean_data = clean(raw_data)
