import gdown
import zipfile
import os
from pathlib import Path


g = "1x8VTx9XGUvCXbiwZnThy88cEbJnSoFOm"


def download_data_g(output_dir):
    """
    Download dataset from Google Drive, extract it, and remove the zip file.
    Args:
        output_dir: Directory where to save and extract the dataset
    Returns:
        Path to the extracted dataset directory
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    zip_path = output_dir / "stanford_cars_dataset_upd2.zip"
    
    url = f"https://drive.google.com/uc?id={g}"
    gdown.download(url, str(zip_path), quiet=False)
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(output_dir)
    
    os.remove(zip_path)