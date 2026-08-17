"""Phase 9 — End-to-end evaluation harness.

Runs the real VIGIL pipeline (S1 Clean -> S2 Detect [YOLOv8] -> S3 Validate
-> S4 Adjudicate) over a labeled image dataset and reports:

  * detection precision / recall / F1 @ IoU 0.5, overall and restricted to
    the security-relevant class subset,
  * per-stage latency (detect, adjudicate),
  * S4 risk-score distribution and the live-LLM vs heuristic-fallback rate.

Dataset: COCO128 (ultralytics.com/assets/coco128.zip), a 128-image subset
of COCO val2017 with YOLO-format labels. It is not the full COCO benchmark;
it is used here as a fast, standard, reproducible smoke-eval subset.

Usage:
    python tools/evaluate.py --images datasets/coco128/images/train2017 \
                              --labels datasets/coco128/labels/train2017 \
                              --limit 128 \
                              --out eval_results.json

If FREELLMAPI_BASE_URL is set in the environment, S4 calls the real
freellmapi gateway; otherwise it degrades to the deterministic heuristic
(as the pipeline is designed to) and the report says so explicitly.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.adjudicator import Adjudicator  # noqa: E402
from agent.freellmapi_client import FreeLlmApiProvider  # noqa: E402
from engines.blocks import CleanBlock, DetectBlock, ValidateBlock  # noqa: E402
from engines.types import Frame  # noqa: E402
from vision.yolo_detector import SECURITY_RELEVANT_CLASSES, YoloDetector  # noqa: E402


def load_yolo_labels(path: Path, names: dict[int, str], img_w: int, img_h: int):
    """Parse a YOLO-format label file into (label, x1, y1, x2, y2) pixel boxes."""
    boxes = []
    if not path.exists():
        return boxes
    for line in path.read_text().strip().splitlines():
        cls_id, xc, yc, w, h = line.split()
        cls_id = int(cls_id)
        xc, yc, w, h = (float(v) for v in (xc, yc, w, h))
        x1 = (xc - w / 2) * img_w
        y1 = (yc - h / 2) * img_h
        x2 = (xc + w / 2) * img_w
        y2 = (yc + h / 2) * img_h
        boxes.append((names.get(cls_id, str(cls_id)), x1, y1, x2, y2))
    return boxes


def iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def match(preds, gts, iou_thresh=0.5):
    """Greedy IoU matching. Returns (tp, fp, fn) counts for one image."""
    matched_gt = set()
    tp = 0
    for label, x1, y1, x2, y2 in preds:
        best_iou, best_j = 0.0, -1
        for j, (glabel, gx1, gy1, gx2, gy2) in enumerate(gts):
            if j in matched_gt or glabel != label:
                continue
            score = iou((x1, y1, x2, y2), (gx1, gy1, gx2, gy2))
            if score > best_iou:
                best_iou, best_j = score, j
        if best_iou >= iou_thresh:
            matched_gt.add(best_j)
            tp += 1
    fp = len(preds) - tp
    fn = len(gts) - len(matched_gt)
    return tp, fp, fn


def run(images_dir: Path, labels_dir: Path, limit: int, out_path: Path) -> dict:
    import cv2  # lazy import, only needed for eval, not the core graph

    detector = YoloDetector(conf=0.25)
    names = detector.names

    provider = None
    live_llm = bool(os.getenv("FREELLMAPI_BASE_URL"))
    if live_llm:
        provider = FreeLlmApiProvider()
    adjudicator = Adjudicator(provider=provider)

    clean = CleanBlock(size=640)
    detect = DetectBlock(detector=detector)
    validate = ValidateBlock(min_confidence=0.25)

    image_paths = sorted(images_dir.glob("*.jpg"))[:limit]

    totals = {"all": [0, 0, 0], "security": [0, 0, 0]}  # tp, fp, fn
    detect_latencies, adjudicate_latencies = [], []
    risks = []
    fallback_count = 0
    n = 0

    for img_path in image_paths:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        n += 1

        frame = Frame(index=n, timestamp=time.time(), width=w, height=h, data=img, source=str(img_path))
        normalized = clean.run({"frame": frame}).outputs["frame"]

        t0 = time.perf_counter()
        detections = detect.run({"frame": normalized}).outputs["detections"]
        detect_latencies.append(time.perf_counter() - t0)

        validated = validate.run({"detections": detections}).outputs["detections"]

        t0 = time.perf_counter()
        decision = adjudicator.decide(validated, context={"scene": "coco128 evaluation frame"})
        adjudicate_latencies.append(time.perf_counter() - t0)
        risks.append(decision.event.risk)
        if decision.used_fallback:
            fallback_count += 1

        gt_path = labels_dir / (img_path.stem + ".txt")
        gts = load_yolo_labels(gt_path, names, w, h)
        preds = [(d.label, *d.bbox) for d in validated.items]

        tp, fp, fn = match(preds, gts)
        totals["all"][0] += tp
        totals["all"][1] += fp
        totals["all"][2] += fn

        sec_preds = [p for p in preds if p[0] in SECURITY_RELEVANT_CLASSES]
        sec_gts = [g for g in gts if g[0] in SECURITY_RELEVANT_CLASSES]
        tp, fp, fn = match(sec_preds, sec_gts)
        totals["security"][0] += tp
        totals["security"][1] += fp
        totals["security"][2] += fn

    def prf(tp, fp, fn):
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        return round(precision, 4), round(recall, 4), round(f1, 4)

    p_all, r_all, f1_all = prf(*totals["all"])
    p_sec, r_sec, f1_sec = prf(*totals["security"])

    report = {
        "dataset": "coco128 (ultralytics.com/assets/coco128.zip)",
        "n_images": n,
        "detector": "yolov8n.pt (COCO-pretrained, conf>=0.25)",
        "reasoning_backend": "freellmapi (live)" if live_llm else "heuristic fallback (no FREELLMAPI_BASE_URL set)",
        "fallback_rate": round(fallback_count / n, 4) if n else None,
        "detection_all_classes": {
            "precision": p_all, "recall": r_all, "f1": f1_all,
            "tp": totals["all"][0], "fp": totals["all"][1], "fn": totals["all"][2],
        },
        "detection_security_relevant": {
            "precision": p_sec, "recall": r_sec, "f1": f1_sec,
            "tp": totals["security"][0], "fp": totals["security"][1], "fn": totals["security"][2],
            "classes": sorted(SECURITY_RELEVANT_CLASSES),
        },
        "latency_ms": {
            "detect_p50": round(sorted(detect_latencies)[len(detect_latencies) // 2] * 1000, 2) if detect_latencies else None,
            "detect_mean": round(sum(detect_latencies) / len(detect_latencies) * 1000, 2) if detect_latencies else None,
            "adjudicate_p50": round(sorted(adjudicate_latencies)[len(adjudicate_latencies) // 2] * 1000, 2) if adjudicate_latencies else None,
            "adjudicate_mean": round(sum(adjudicate_latencies) / len(adjudicate_latencies) * 1000, 2) if adjudicate_latencies else None,
        },
        "risk_score": {
            "mean": round(sum(risks) / len(risks), 4) if risks else None,
            "max": round(max(risks), 4) if risks else None,
            "min": round(min(risks), 4) if risks else None,
        },
    }

    out_path.write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the VIGIL pipeline end to end.")
    parser.add_argument("--images", type=Path, default=Path("datasets/coco128/images/train2017"))
    parser.add_argument("--labels", type=Path, default=Path("datasets/coco128/labels/train2017"))
    parser.add_argument("--limit", type=int, default=128)
    parser.add_argument("--out", type=Path, default=Path("eval_results.json"))
    args = parser.parse_args()

    result = run(args.images, args.labels, args.limit, args.out)
    print(json.dumps(result, indent=2))
