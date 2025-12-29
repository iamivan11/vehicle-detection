import lightning as L
from ultralytics import YOLO

class YOLODetector(L.LightningModule):
    def __init__(self):
        super().__init__()
        yolo = YOLO()