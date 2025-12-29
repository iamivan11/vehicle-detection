from pathlib import Path

import fire
from hydra import compose, initialize_config_dir


def get_config(overrides: list[str] | None = None):
    """Load Hydra configuration."""
    config_dir = Path(__file__).parent.parent / "configs"
    with initialize_config_dir(version_base=None, config_dir=str(config_dir.absolute())):
        return compose(config_name="config", overrides=overrides or [])


def train(
    batch_size: int | None = None,
    lr: float | None = None,
    max_epochs: int | None = None,
    **kwargs,
) -> None:
    """Train the vehicle detection model.

    Args:
        batch_size: Training batch size
        lr: Learning rate
        max_epochs: Maximum training epochs
        **kwargs: Additional Hydra overrides
    """
    overrides = []
    if batch_size is not None:
        overrides.append(f"training.batch_size={batch_size}")
    if lr is not None:
        overrides.append(f"training.lr={lr}")
    if max_epochs is not None:
        overrides.append(f"training.max_epochs={max_epochs}")

    for key, value in kwargs.items():
        overrides.append(f"{key}={value}")

    cfg = get_config(overrides=overrides)

    from vehicle_detection.train import train as run_train

    run_train(cfg)


def infer(
    image: str | None = None,
    image_dir: str | None = None,
    output_dir: str = "results",
    checkpoint: str = "checkpoints/last.ckpt",
    score_threshold: float = 0.5,
) -> None:
    """Run inference on images.

    Args:
        image: Path to single image
        image_dir: Path to directory of images
        output_dir: Output directory for results
        checkpoint: Path to model checkpoint
        score_threshold: Minimum confidence threshold
    """
    import json

    cfg = get_config()

    from vehicle_detection.data import ensure_data_exists
    from vehicle_detection.infer import infer_batch, infer_single, load_model

    ensure_data_exists(Path.cwd())

    model = load_model(checkpoint, cfg)

    if image:
        result = infer_single(model, image, cfg.data.image_size, score_threshold)
        print(json.dumps(result, indent=2))
    elif image_dir:
        infer_batch(model, image_dir, output_dir, cfg.data.image_size, score_threshold)
    else:
        print("Please provide --image or --image-dir")


def download() -> None:
    """Download training data from Google Drive."""
    from vehicle_detection.data import download_data

    download_data(Path.cwd())


def main() -> None:
    """Main CLI entry point."""
    fire.Fire(
        {
            "train": train,
            "infer": infer,
            "download": download,
        }
    )


if __name__ == "__main__":
    main()
