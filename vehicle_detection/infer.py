import json
import logging
from pathlib import Path
from typing import Any

import hydra
import torch
from omegaconf import DictConfig
from PIL import Image
from torchvision import transforms

from vehicle_detection.data import ensure_data_exists
from vehicle_detection.models import VehicleDetector

logger = logging.getLogger(__name__)


def load_model(checkpoint_path: str, cfg: DictConfig) -> VehicleDetector:
    """Load trained model from checkpoint."""
    model = VehicleDetector.load_from_checkpoint(
        checkpoint_path,
        num_classes=cfg.data.num_classes + 1,
        box_score_thresh=cfg.model.box_score_thresh,
        box_nms_thresh=cfg.model.box_nms_thresh,
    )
    model.eval()
    return model


def preprocess_image(image_path: str | Path, image_size: int) -> torch.Tensor:
    """Load and preprocess image for inference."""
    image = Image.open(image_path).convert("RGB")

    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    return transform(image)


def format_predictions(
    predictions: dict,
    class_names: list[str],
    score_threshold: float,
) -> dict[str, Any]:
    """Format model predictions to output JSON structure."""
    boxes = predictions["boxes"].cpu().numpy()
    scores = predictions["scores"].cpu().numpy()
    labels = predictions["labels"].cpu().numpy()

    detections = []
    for box, score, label in zip(boxes, scores, labels, strict=True):
        if score >= score_threshold:
            detections.append(
                {
                    "bbox": [float(x) for x in box],
                    "score": float(score),
                    "class_id": int(label),
                    "class_name": class_names[label] if label < len(class_names) else "unknown",
                }
            )

    return {"detections": detections}


def infer_single(
    model: VehicleDetector,
    image_path: str | Path,
    class_names: list[str],
    image_size: int,
    score_threshold: float,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> dict[str, Any]:
    """Run inference on a single image."""
    model = model.to(device)
    image = preprocess_image(image_path, image_size).to(device)

    with torch.no_grad():
        predictions = model([image])[0]

    return format_predictions(predictions, class_names, score_threshold)


def infer_batch(
    model: VehicleDetector,
    image_dir: str | Path,
    output_dir: str | Path,
    class_names: list[str],
    image_size: int,
    score_threshold: float,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> None:
    """Run inference on a directory of images."""
    image_dir = Path(image_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = model.to(device)
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}

    image_files = [f for f in image_dir.iterdir() if f.suffix.lower() in image_extensions]

    logger.info(f"Processing {len(image_files)} images...")

    for image_path in image_files:
        result = infer_single(model, image_path, class_names, image_size, score_threshold, device)

        output_path = output_dir / f"{image_path.stem}.json"
        with output_path.open("w") as f:
            json.dump(result, f, indent=2)

        logger.info(f"Processed {image_path.name} -> {output_path.name}")


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main inference entry point."""
    import fire

    def run_inference(
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
        project_root = Path(hydra.utils.get_original_cwd())
        ensure_data_exists(project_root)

        class_names = ["background", *cfg.data.class_names]

        logger.info(f"Loading model from {checkpoint}")
        model = load_model(checkpoint, cfg)

        if image:
            result = infer_single(
                model,
                image,
                class_names,
                cfg.data.image_size,
                score_threshold,
            )
            print(json.dumps(result, indent=2))
        elif image_dir:
            infer_batch(
                model,
                image_dir,
                output_dir,
                class_names,
                cfg.data.image_size,
                score_threshold,
            )
        else:
            logger.error("Please provide --image or --image-dir")

    fire.Fire(run_inference)


if __name__ == "__main__":
    main()
