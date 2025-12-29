from pathlib import Path

from vehicle_detection.data import download_data


def main() -> None:
    """Download dataset to project root."""
    print("Hello from vehicle-detection!")
    print("Downloading the data...")
    download_data(output_dir=Path(__file__).parent)
    print("Successfully downloaded!")


if __name__ == "__main__":
    main()
