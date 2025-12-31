import logging
import zipfile
from pathlib import Path

import gdown

from vehicle_detection.constants import DATASET_FOLDER, GDRIVE_FILE_ID

logger = logging.getLogger(__name__)


def pull_dvc_data() -> bool:
    """Try to pull data using DVC.

    Returns:
        True if DVC pull succeeded, False otherwise.
    """
    try:
        from dvc.repo import Repo
        
        logger.info("Attempting DVC pull...")
        repo = Repo()
        repo.pull()
        logger.info("DVC pull succeeded")
        return True
    except ImportError:
        logger.warning("DVC not installed")
        return False
    except Exception as e:
        logger.warning(f"DVC pull failed: {e}")
        return False


def download_data(output_dir: str | Path = ".") -> Path:
    """Download dataset from Google Drive, extract it, and remove the zip file.

    Args:
        output_dir: Directory where to save and extract the dataset

    Returns:
        Path to the extracted dataset directory
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = output_dir / DATASET_FOLDER
    if dataset_path.exists():
        logger.info(f"Dataset already exists at {dataset_path}")
        return dataset_path

    zip_path = output_dir / "stanford_cars_dataset_upd2.zip"
    url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"

    logger.info("Downloading dataset from Google Drive...")
    gdown.download(url, str(zip_path), quiet=False)

    logger.info(f"Extracting {zip_path}...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(output_dir)

    zip_path.unlink()
    logger.info(f"Dataset extracted to {dataset_path}")

    return dataset_path


def ensure_data_exists(data_dir: str | Path = ".") -> Path:
    """Ensure dataset exists: try DVC first, then download.

    Args:
        data_dir: Directory containing or to contain the dataset

    Returns:
        Path to the dataset directory
    """
    data_dir = Path(data_dir)
    dataset_path = data_dir / DATASET_FOLDER

    if dataset_path.exists():
        logger.info(f"Dataset found at {dataset_path}")
        return dataset_path

    logger.info("Dataset not found, attempting to fetch...")

    if pull_dvc_data() and dataset_path.exists():
        return dataset_path

    logger.info("Falling back to direct download...")
    download_data(data_dir)

    return dataset_path
