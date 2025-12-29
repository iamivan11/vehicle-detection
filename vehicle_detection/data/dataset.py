from pathlib import Path

import albumentations as A
import numpy as np
import pytorch_lightning as pl
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import DataLoader, Dataset


class VehicleDataset(Dataset):
    """Dataset for vehicle detection with YOLO format annotations."""

    def __init__(
        self,
        images_dir: Path,
        labels_dir: Path,
        transforms: A.Compose | None = None,
    ) -> None:
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.transforms = transforms

        self.image_files = sorted(
            [f for f in self.images_dir.iterdir() if f.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        )

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        image_path = self.image_files[idx]
        label_path = self.labels_dir / f"{image_path.stem}.txt"

        image = Image.open(image_path).convert("RGB")
        image_array = np.array(image)
        img_h, img_w = image_array.shape[:2]

        boxes, labels = self._parse_yolo_labels(label_path, img_w, img_h)

        if self.transforms:
            transformed = self.transforms(
                image=image_array,
                bboxes=boxes,
                labels=labels,
            )
            image_array = transformed["image"]
            boxes = transformed["bboxes"]
            labels = transformed["labels"]

        if len(boxes) > 0:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx]),
        }

        return image_array, target

    def _parse_yolo_labels(
        self, label_path: Path, img_w: int, img_h: int
    ) -> tuple[list[list[float]], list[int]]:
        """Parse YOLO format labels and convert to pascal_voc format.

        YOLO format: class_id x_center y_center width height (normalized 0-1)
        Pascal VOC format: x_min y_min x_max y_max (absolute pixels)
        """
        boxes = []
        labels = []

        if not label_path.exists():
            return boxes, labels

        with label_path.open() as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue

                class_id = int(parts[0])
                x_center = float(parts[1]) * img_w
                y_center = float(parts[2]) * img_h
                width = float(parts[3]) * img_w
                height = float(parts[4]) * img_h

                x_min = max(0, x_center - width / 2)
                y_min = max(0, y_center - height / 2)
                x_max = min(img_w, x_center + width / 2)
                y_max = min(img_h, y_center + height / 2)

                if x_max > x_min and y_max > y_min:
                    boxes.append([x_min, y_min, x_max, y_max])
                    labels.append(class_id)

        return boxes, labels


def get_train_transforms(image_size: int) -> A.Compose:
    """Get training augmentations."""
    return A.Compose(
        [
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.ColorJitter(p=0.2),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"], min_visibility=0.3),
    )


def get_val_transforms(image_size: int) -> A.Compose:
    """Get validation/test transforms."""
    return A.Compose(
        [
            A.Resize(image_size, image_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"], min_visibility=0.3),
    )


def collate_fn(batch: list) -> tuple[list[torch.Tensor], list[dict]]:
    """Custom collate function for detection."""
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    return images, targets


class VehicleDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for vehicle detection."""

    def __init__(
        self,
        dataset_dir: str | Path,
        batch_size: int = 8,
        num_workers: int = 4,
        image_size: int = 640,
    ) -> None:
        super().__init__()
        self.dataset_dir = Path(dataset_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size

    def setup(self, stage: str | None = None) -> None:
        """Setup is handled in dataloaders since data is already split."""
        pass

    def train_dataloader(self) -> DataLoader:
        dataset = VehicleDataset(
            images_dir=self.dataset_dir / "images" / "train",
            labels_dir=self.dataset_dir / "labels" / "train",
            transforms=get_train_transforms(self.image_size),
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        dataset = VehicleDataset(
            images_dir=self.dataset_dir / "images" / "val",
            labels_dir=self.dataset_dir / "labels" / "val",
            transforms=get_val_transforms(self.image_size),
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        dataset = VehicleDataset(
            images_dir=self.dataset_dir / "images" / "test",
            labels_dir=self.dataset_dir / "labels" / "test",
            transforms=get_val_transforms(self.image_size),
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )
