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
    pl.seed_everything(cfg.train.seed)

    logger.info("Ensuring data exists...")
    try:
        # @hydra.main
        project_root = Path(hydra.utils.get_original_cwd())
    except ValueError:
        # compose()
        project_root = Path.cwd()
    dataset_path = ensure_data_exists(project_root)

    logger.info("Setting up data module...")
    datamodule = VehicleDataModule(
        dataset_dir=dataset_path,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        image_size=cfg.data.image_size,
    )

    logger.info("Creating model...")
    model = VehicleDetector(
        num_classes=cfg.data.num_classes + 1,
        pretrained=cfg.model.pretrained,
        trainable_backbone_layers=cfg.model.trainable_backbone_layers,
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
        max_epochs=cfg.train.max_epochs,
        warmup_epochs=cfg.train.warmup_epochs,
        box_score_thresh=cfg.model.box_score_thresh,
        box_nms_thresh=cfg.model.box_nms_thresh,
    )

    checkpoint_dir = project_root / cfg.paths.checkpoint_dir
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    callbacks = [
        ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename="vehicle-detector-{epoch:02d}-{val_map:.3f}",
            monitor=cfg.train.monitor,
            mode=cfg.train.mode,
            save_top_k=cfg.train.save_top_k,
            save_last=True,
        ),
        EarlyStopping(
            monitor=cfg.train.monitor,
            mode=cfg.train.mode,
            patience=cfg.train.patience,
            min_delta=cfg.train.min_delta,
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
        max_epochs=cfg.train.max_epochs,
        accelerator=cfg.train.accelerator,
        devices=cfg.train.devices,
        precision=cfg.train.precision,
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
        logger.info(f"Best {cfg.train.monitor}: {best_score:.4f}")
        return float(best_score)

    return 0.0


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main entry point for training."""
    logger.info("Configuration:")
    logger.info(OmegaConf.to_yaml(cfg))
    train(cfg)


if __name__ == "__main__":
    main()
