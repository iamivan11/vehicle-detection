# Vehicle Detection

Automatic detection and classification of vehicles in road images using computer
vision and deep learning.

## Project Overview

This project implements a neural network-based system for vehicle detection and
classification. The system:

- **Localizes objects** using bounding boxes
- **Classifies vehicles** by body type (coupe, sedan, truck, SUV, hatchback,
  convertible, minivan, van, wagon)
- **Outputs standardized JSON** for analytics and monitoring

### Input/Output Format

**Input:**

- RGB images (JPG/PNG)
- Arbitrary size (normalized to 640×640)

**Output:**

```json
{
  "detections": [
    {
      "bbox": [x_min, y_min, x_max, y_max],
      "score": 0.95,
      "class_id": 0,
      "class_name": "sedan"
    }
  ]
}
```

### Metrics

- **mAP@0.5** - primary metric (target: 0.80-0.95)
- **mAP@0.5:0.95** - strict metric (target: 0.50-0.70)
- **Precision/Recall** - error analysis
- **Per-class AP** - class-specific performance

### Dataset

**Stanford Cars Dataset:**

- 16,185 images
- 196 original classes → generalized to 9 body types
- Split: 70% train / 15% val / 15% test

### Model Architecture

**Baseline:** YOLOv8n (5-10 epochs, mAP@0.5 ≈ 0.20-0.40)

**Main model:** YOLOv8m/YOLOv11m

- Single-stage detection
- AdamW optimizer with OneCycleLR scheduler
- 30-60 epochs with early stopping

---

## Setup

### Requirements

- Python 3.11+
- uv package manager

### Installation

```bash
# Clone repository
git clone https://github.com/username/vehicle-detection.git
cd vehicle-detection

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create environment and install dependencies
uv venv
source .venv/bin/activate  # Linux/Mac
uv sync

# Setup pre-commit hooks
pre-commit install
pre-commit run -a
```

### Data Setup

```bash
# Pull data with DVC
dvc pull
```

---

## Train

### Quick Start

```bash
uv run python -m vehicle_detection.train
```

### Custom Configuration

```bash
# Change model
uv run python -m vehicle_detection.train model=yolov8m

# Adjust hyperparameters
uv run python -m vehicle_detection.train \
    training.batch_size=32 \
    training.lr=0.001
```

### Monitoring

```bash
# Launch MLflow UI
mlflow ui --host 127.0.0.1 --port 8080
```

Open http://127.0.0.1:8080 in browser.

---

## Inference

```bash
# Single image
uv run python -m vehicle_detection.infer \
    --image path/to/image.jpg \
    --checkpoint path/to/model.ckpt

# Batch inference
uv run python -m vehicle_detection.infer \
    --image-dir path/to/images/ \
    --output-dir results/
```

---

## Project Structure

```
vehicle-detection/
├── configs/                 # Hydra configs
├── vehicle_detection/       # Main package
│   ├── data/               # Data loading
│   ├── models/             # Model definitions
│   ├── train.py            # Training script
│   └── infer.py            # Inference script
├── .pre-commit-config.yaml
├── pyproject.toml
└── README.md
```

---

## Development

### Code Quality

Tools: ruff (formatting & linting), prettier, pre-commit

```bash
pre-commit run -a
```
