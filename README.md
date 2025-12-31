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
      "bbox": [100, 150, 300, 400],
      "score": 0.95,
      "class_id": 1,
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
- Split: 50% train / 25% val / 25% test

### Model Architecture

**Model:** Faster R-CNN with ResNet50-FPN backbone

- Two-stage detection (region proposals + classification)
- AdamW optimizer with Cosine Annealing scheduler (with warmup)

#### Baseline
- Train for 2-5 epochs
- Expected mAP@0.5: ~0.15–0.30

#### Main
- Train for 30 epochs
- Expected mAP@0.5: ~0.70-0.85

## Setup

### Requirements

- Python 3.12+
- uv package manager

### Installation

```bash
# Clone repository
git clone https://github.com/iamivan11/vehicle-detection.git
cd vehicle-detection

# Install uv (if not installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create environment and install dependencies
uv venv
source .venv/bin/activate  # Linux/Mac
#.venv\Scripts\activate  # Windows

uv sync

# Setup pre-commit hooks
pre-commit install
pre-commit run -a
```

## Train

### Quick Start

```bash
uv run python -m vehicle_detection.commands train
```

### Custom Configuration

```bash
# Adjust hyperparameters
uv run python -m vehicle_detection.commands train \
    --batch_size=8 \
    --max_epochs=5 \
    --precision=32 \
    --accelerator=gpu \
    --num_workers=2 \
    --tracking_uri=http://127.0.0.1:8080
```

### Monitoring

```bash
# Launch MLflow UI (in separate terminal)
mlflow ui --host 127.0.0.1 --port 8080
```

Open <http://127.0.0.1:8080> in browser to view training metrics.

## Inference

```bash
# Single image
uv run python -m vehicle_detection.commands infer \
    --image path/to/image.jpg \
    --checkpoint checkpoints/best.ckpt

# Batch inference
uv run python -m vehicle_detection.commands infer \
    --image-dir path/to/images/ \
    --output-dir results/
```

## Project Structure

```
vehicle-detection/
├── .dvc/                     # DVC configuration
├── configs/                  # Hydra configs
│   ├── config.yaml           # Main config
│   ├── model/                # Model configs
│   ├── training/             # Training configs
│   └── data/                 # Data configs
├── vehicle_detection/        # Main package
│   ├── data/                 # Data loading
│   │   ├── dataset.py        # Dataset and DataModule
│   │   └── download.py       # Data download utilities
│   ├── models/               # Model definitions
│   │   └── detector.py       # Lightning Module
│   ├── train.py              # Training script
│   ├── infer.py              # Inference script
│   ├── constants.py          # Constants for download.py
│   └── commands.py           # CLI entry point
├── .pre-commit-config.yaml
├── pyproject.toml
├── uv.lock
└── README.md
```

## Development

### Code Quality

Tools: ruff (formatting & linting), prettier, pre-commit

```bash
# Run all checks
pre-commit run -a

# Format code
ruff format .

# Lint code
ruff check . --fix
```
