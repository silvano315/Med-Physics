import kaggle
from pathlib import Path

def download_kaggle_dataset(dataset_name : str = None, kaggle_url : str = None):
    """
    Download ACDC dataset from Kaggle using Kaggle API.
    Requires:
    1. Kaggle account
    2. API token (kaggle.json) in ~/.kaggle/
    3. kaggle package installed: pip install kaggle
    """

    data_dir = Path('data')
    data_dir.mkdir(exist_ok=True)

    print(f"Downloading {dataset_name} dataset from Kaggle...")
    try:
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(
            kaggle_url,
            path=data_dir,
            unzip=True
        )
        print("Dataset downloaded and extracted successfully!")

    except Exception as e:
        print(f"Error downloading dataset: {e}")
        print("\nPlease ensure you have:")
        print("1. Created a Kaggle account")
        print("2. Generated an API token from https://www.kaggle.com/settings")
        print("3. Placed kaggle.json in ~/.kaggle/")
        print("4. Set appropriate permissions: chmod 600 ~/.kaggle/kaggle.json")
        raise