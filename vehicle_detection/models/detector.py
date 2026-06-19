from typing import Any

import pytorch_lightning as pl
import torch
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.models import ResNet50_Weights
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_Weights,
    fasterrcnn_resnet50_fpn,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

MODELS = {
    "fasterrcnn_resnet50_fpn": (
        fasterrcnn_resnet50_fpn,
        FasterRCNN_ResNet50_FPN_Weights,
        ResNet50_Weights,
    ),
}


class VehicleDetector(pl.LightningModule):
    """Lightning module wrapping Faster R-CNN for vehicle detection."""

    def __init__(
        self,
        model_name: str = "fasterrcnn_resnet50_fpn",
        num_classes: int = 10,
        pretrained: bool = True,
        pretrained_backbone: bool = True,
        trainable_backbone_layers: int = 3,
        lr: float = 0.001,
        weight_decay: float = 0.0005,
        max_epochs: int = 30,
        warmup_epochs: int = 3,
        box_score_thresh: float = 0.05,
        box_nms_thresh: float = 0.5,
        box_detections_per_img: int = 100,
        class_names: list[str] | None = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.lr = lr
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.warmup_epochs = warmup_epochs
        self.class_names = class_names

        if model_name not in MODELS:
            raise ValueError(f"Unknown model '{model_name}'. Available: {list(MODELS)}")
        build_fn, weights_enum, backbone_weights_enum = MODELS[model_name]

        weights = weights_enum.DEFAULT if pretrained else None
        weights_backbone = (
            backbone_weights_enum.DEFAULT if pretrained_backbone and not pretrained else None
        )
        self.model = build_fn(
            weights=weights,
            weights_backbone=weights_backbone,
            trainable_backbone_layers=trainable_backbone_layers,
            box_score_thresh=box_score_thresh,
            box_nms_thresh=box_nms_thresh,
            box_detections_per_img=box_detections_per_img,
        )

        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        self.val_map = MeanAveragePrecision(iou_type="bbox", class_metrics=True)

    def forward(self, images: list[Tensor], targets: list[dict] | None = None) -> Any:
        return self.model(images, targets)

    def training_step(self, batch: tuple, batch_idx: int) -> Tensor:
        images, targets = batch

        loss_dict = self.model(images, targets)

        loss_classifier = loss_dict.get("loss_classifier", torch.tensor(0.0))
        loss_box_reg = loss_dict.get("loss_box_reg", torch.tensor(0.0))
        loss_objectness = loss_dict.get("loss_objectness", torch.tensor(0.0))
        loss_rpn_box_reg = loss_dict.get("loss_rpn_box_reg", torch.tensor(0.0))

        total_loss = loss_classifier + loss_box_reg + loss_objectness + loss_rpn_box_reg

        self.log("train_loss", total_loss, prog_bar=True, batch_size=len(images))
        self.log("train_loss_classifier", loss_classifier, batch_size=len(images))
        self.log("train_loss_box_reg", loss_box_reg, batch_size=len(images))
        self.log("train_loss_objectness", loss_objectness, batch_size=len(images))
        self.log("train_loss_rpn_box_reg", loss_rpn_box_reg, batch_size=len(images))

        return total_loss

    def validation_step(self, batch: tuple, batch_idx: int) -> None:
        images, targets = batch

        self.model.train()
        with torch.no_grad():
            loss_dict = self.model(images, targets)
        total_loss = sum(loss_dict.values())
        self.log("val_loss", total_loss, prog_bar=True, batch_size=len(images))

        self.model.eval()
        predictions = self.model(images)

        preds = [
            {
                "boxes": pred["boxes"],
                "scores": pred["scores"],
                "labels": pred["labels"],
            }
            for pred in predictions
        ]
        target_formatted = [
            {
                "boxes": t["boxes"],
                "labels": t["labels"],
            }
            for t in targets
        ]

        self.val_map.update(preds, target_formatted)

    def on_validation_epoch_end(self) -> None:
        map_results = self.val_map.compute()

        self.log("val_map", map_results["map"], prog_bar=True)
        self.log("val_map_50", map_results["map_50"])
        self.log("val_map_75", map_results["map_75"])
        self.log("val_mar_100", map_results["mar_100"])

        per_class = map_results.get("map_per_class")
        classes = map_results.get("classes")
        if per_class is not None and per_class.ndim > 0:
            for cls_id, ap in zip(classes.tolist(), per_class.tolist(), strict=False):
                if self.class_names and 1 <= cls_id <= len(self.class_names):
                    label = self.class_names[cls_id - 1]
                else:
                    label = str(cls_id)
                self.log(f"val_ap_{label}", ap)

        self.val_map.reset()

    def predict_step(self, batch: tuple, batch_idx: int) -> list[dict]:
        images, _ = batch
        self.model.eval()
        return self.model(images)

    def configure_optimizers(self) -> dict:
        optimizer = AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=self.warmup_epochs,
        )

        main_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.max_epochs - self.warmup_epochs,
            eta_min=self.lr * 0.01,
        )

        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[self.warmup_epochs],
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }
