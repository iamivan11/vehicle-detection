import logging
import subprocess
from pathlib import Path

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger

from vehicle_detection.data import VehicleDataModule, ensure_data_exists
from vehicle_detection.models import VehicleDetector

logger = logging.getLogger(__name__)

DATASET_FOLDER = "stanford_cars_dataset"


def get_git_commit_id() -> str:
    """Get current git commit ID."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()[:8]
    except Exception:
        return "unknown"


def train(cfg: DictConfig) -> float:
    """Run training with the given configuration."""
    pl.seed_everything(cfg.training.seed)

    logger.info("Ensuring data exists...")
    project_root = Path(hydra.utils.get_original_cwd())
    dataset_path = ensure_data_exists(project_root)

    logger.info("Setting up data module...")
    datamodule = VehicleDataModule(
        dataset_dir=dataset_path,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.training.num_workers,
        image_size=cfg.data.image_size,
    )

    logger.info("Creating model...")
    model = VehicleDetector(
        num_classes=cfg.data.num_classes + 1,  # +1 for background
        pretrained=cfg.model.pretrained,
        trainable_backbone_layers=cfg.model.trainable_backbone_layers,
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
        max_epochs=cfg.training.max_epochs,
        warmup_epochs=cfg.training.warmup_epochs,
        box_score_thresh=cfg.model.box_score_thresh,
        box_nms_thresh=cfg.model.box_nms_thresh,
    )

    checkpoint_dir = project_root / cfg.paths.checkpoint_dir
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    callbacks = [
        ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename="vehicle-detector-{epoch:02d}-{val_map:.3f}",
            monitor=cfg.training.monitor,
            mode=cfg.training.mode,
            save_top_k=cfg.training.save_top_k,
            save_last=True,
        ),
        EarlyStopping(
            monitor=cfg.training.monitor,
            mode=cfg.training.mode,
            patience=cfg.training.patience,
            min_delta=cfg.training.min_delta,
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    mlflow_logger = MLFlowLogger(
        experiment_name=cfg.mlflow.experiment_name,
        tracking_uri=cfg.mlflow.tracking_uri,
        log_model=True,
    )

    mlflow_logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))

    git_commit = get_git_commit_id()
    mlflow_logger.experiment.log_param(mlflow_logger.run_id, "git_commit", git_commit)

    logger.info("Starting training...")
    trainer = pl.Trainer(
        max_epochs=cfg.training.max_epochs,
        accelerator=cfg.training.accelerator,
        devices=cfg.training.devices,
        precision=cfg.training.precision,
        callbacks=callbacks,
        logger=mlflow_logger,
        log_every_n_steps=10,
        enable_progress_bar=True,
    )

    trainer.fit(model, datamodule)

    best_model_path = callbacks[0].best_model_path
    logger.info(f"Best model saved to: {best_model_path}")

    best_score = callbacks[0].best_model_score
    if best_score is not None:
        logger.info(f"Best {cfg.training.monitor}: {best_score:.4f}")
        return float(best_score)

    return 0.0


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main entry point for training."""
    logger.info("Configuration:")
    logger.info(OmegaConf.to_yaml(cfg))
    train(cfg)


if __name__ == "__main__":
    main()