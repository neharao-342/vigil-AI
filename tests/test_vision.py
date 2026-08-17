"""Phase 9 tests for the L3 real detector backend (vision/yolo_detector.py).

Wires the previously-stub S2 Detect block to a real YOLOv8 model. Model
weights are downloaded on first use (~6MB); tests skip cleanly if that
download can't happen (no network / offline CI runner) rather than failing
the whole suite, matching the project's "importable without GPU deps"
promise.
"""
from __future__ import annotations

import pytest

from engines.blocks import DetectBlock
from engines.types import NormalizedFrame

ultralytics = pytest.importorskip("ultralytics", reason="ultralytics not installed")


@pytest.fixture(scope="module")
def detector():
    from vision.yolo_detector import YoloDetector

    try:
        return YoloDetector(conf=0.25)
    except Exception as exc:  # noqa: BLE001 - offline/no weights cache
        pytest.skip(f"could not load yolov8n weights: {exc}")


def test_security_relevant_classes_are_coco_labels(detector):
    from vision.yolo_detector import SECURITY_RELEVANT_CLASSES

    coco_names = set(detector.names.values())
    assert SECURITY_RELEVANT_CLASSES.issubset(coco_names)


def test_yolo_detector_on_blank_frame_returns_no_detections(detector):
    import numpy as np

    blank = np.zeros((640, 640, 3), dtype="uint8")
    frame = NormalizedFrame(index=0, timestamp=0.0, width=640, height=640, data=blank)
    detections = list(detector(frame))
    assert detections == []


def test_detect_block_with_real_detector_stays_within_declared_contract(detector):
    import numpy as np

    blank = np.zeros((640, 640, 3), dtype="uint8")
    frame = NormalizedFrame(index=1, timestamp=0.0, width=640, height=640, data=blank)
    result = DetectBlock(detector=detector).run({"frame": frame})
    detections = result.outputs["detections"]
    assert detections.frame_index == 1
    assert isinstance(detections.items, list)
