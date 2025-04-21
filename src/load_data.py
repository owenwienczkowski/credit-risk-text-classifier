import kagglehub
from pathlib import Path
import shutil

# Download latest version
def download_raw_data():
    # Locate dataset
    path = kagglehub.dataset_download("laotse/credit-risk-dataset")
    # Find all .csv files (should only be 1)
    data = list(Path(path).glob("*.csv"))
    if not data:
        raise FileNotFoundError("No CSV file found in downloaded dataset.")
    source_file = data[0]

    # Copy the dataset to the target location
    target = Path("data/raw/credit_risk_dataset.csv")
    if target.exists():
        print("Overwriting existing raw data...")  

    # Write data into target location
    shutil.copy(source_file,target)
    
    return target