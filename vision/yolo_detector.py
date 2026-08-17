"""L3 — YOLOv8 detector backend.

Fills the previously-stub S2 Detect block with a real object detector.
Exposes a callable matching the contract `engines.blocks.DetectBlock`
already declares (`config['detector']: Callable[[NormalizedFrame],
Iterable[Detection]]`), so no change to the block itself is required —
this is pure dependency injection, keeping the L3 graph importable and
GPU-free when the detector isn't wired in (tests still exercise the stub
path unmodified).

Weights are pulled once (COCO-pretrained yolov8n.pt, ~6MB) and cached.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Iterable

from engines.types import Detection, NormalizedFrame

# Classes VIGIL treats as security-relevant for the perimeter/anomaly use
# case the README describes. Detection still runs over all 80 COCO classes;
# this set is used only to slice out a relevant subset for reporting.
SECURITY_RELEVANT_CLASSES = {
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "bus",
    "truck",
    "backpack",
    "handbag",
    "suitcase",
    "knife",
    "dog",
}


@lru_cache(maxsize=1)
def _load_model(weights: str):
    from ultralytics import YOLO  # imported lazily: heavy, optional dep

    return YOLO(weights)


class YoloDetector:
    """ReasoningProvider-style detector: COCO-pretrained YOLOv8."""

    def __init__(
        self,
        weights: str = "vision/models/yolov8n.pt",
        conf: float = 0.25,
        classes: set[str] | None = None,
    ) -> None:
        self.model = _load_model(weights)
        self.conf = conf
        self.names = self.model.names
        self.classes = classes

    def __call__(self, frame: NormalizedFrame) -> Iterable[Detection]:
        if frame.data is None:
            return []
        results = self.model.predict(frame.data, conf=self.conf, verbose=False)
        result = results[0]
        detections: list[Detection] = []
        for box in result.boxes:
            label = self.names[int(box.cls[0])]
            if self.classes is not None and label not in self.classes:
                continue
            x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
            detections.append(
                Detection(
                    label=label,
                    confidence=float(box.conf[0]),
                    bbox=(x1, y1, x2, y2),
                )
            )
        return detections
