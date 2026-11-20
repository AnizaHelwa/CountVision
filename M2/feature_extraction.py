import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from skimage.feature import local_binary_pattern

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))

from grape_pipeline import detect_grapes
from preprocess import preprocess, FINAL_CONFIG

data_dir = root / "yolo_variants" / "P0"
result_dir = Path(__file__).resolve().parent / "results"
result_dir.mkdir(exist_ok=True)


def read_labels(path, width, height):
    boxes = []

    if not path.exists():
        raise FileNotFoundError(f"Label tidak ditemukan: {path}")

    for line in path.read_text().splitlines():
        if not line.strip():
            continue

        cls, xc, yc, bw, bh = map(float, line.split())

        if int(cls) != 0:
            continue

        x1 = (xc - bw / 2) * width
        y1 = (yc - bh / 2) * height
        x2 = (xc + bw / 2) * width
        y2 = (yc + bh / 2) * height

        boxes.append((x1, y1, x2, y2))

    return boxes


def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])

    return intersection / (area1 + area2 - intersection + 1e-8)


def extract_features(img, mask, box):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = map(int, box)

    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    roi = img[y1:y2, x1:x2]
    roi_mask = mask[y1:y2, x1:x2]

    if roi.size == 0:
        return None

    binary = (roi_mask > 0).astype(np.uint8)
    if binary.sum() == 0:
        return None

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    pixels = hsv[binary > 0]

    mean_hsv = pixels.mean(axis=0)
    std_hsv = pixels.std(axis=0)

    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    area = float(binary.sum())
    perimeter = sum(cv2.arcLength(c, True) for c in contours)

    circularity = 4 * np.pi * area / (perimeter ** 2 + 1e-8)
    aspect_ratio = (x2 - x1) / max(y2 - y1, 1)
    fill_ratio = area / max((x2-x1) * (y2-y1), 1)

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, 8, 1, method="uniform")

    values = lbp[binary > 0]
    hist, _ = np.histogram(values, bins=np.arange(11))
    hist = hist / (hist.sum() + 1e-8)

    features = {
        "mean_h": mean_hsv[0],
        "mean_s": mean_hsv[1],
        "mean_v": mean_hsv[2],
        "std_h": std_hsv[0],
        "std_s": std_hsv[1],
        "std_v": std_hsv[2],
        "area": area / (w * h),
        "perimeter": perimeter / max(w + h, 1),
        "circularity": circularity,
        "aspect_ratio": aspect_ratio,
        "fill_ratio": fill_ratio
    }

    for i, value in enumerate(hist):
        features[f"lbp_{i}"] = value

    return features


rows = []
stats = []

for split in ["train", "valid", "test"]:
    image_dir = data_dir / split / "images"
    label_dir = data_dir / split / "labels"

    image_paths = sorted(
        p for p in image_dir.iterdir()
        if p.suffix.lower() in [".jpg", ".jpeg", ".png"]
    )

    total_candidates = 0
    total_gt = 0

    for path in image_paths:
        img = cv2.imread(str(path))

        if img is None:
            print("Gagal membaca:", path.name)
            continue

        boxes, mask = detect_grapes(img)
        processed = preprocess(img, FINAL_CONFIG)

        h, w = processed.shape[:2]
        gt_boxes = read_labels(
            label_dir / f"{path.stem}.txt", w, h
        )

        total_gt += len(gt_boxes)
        total_candidates += len(boxes)

        for i, box in enumerate(boxes):
            features = extract_features(processed, mask, box)

            if features is None:
                continue

            max_iou = max(
                (calculate_iou(box, gt) for gt in gt_boxes),
                default=0
            )

            if max_iou >= 0.5:
                label = 1
            elif max_iou <= 0.1:
                label = 0
            else:
                label = -1

            rows.append({
                "filename": path.name,
                "split": split,
                "candidate_id": i,
                "x1": box[0],
                "y1": box[1],
                "x2": box[2],
                "y2": box[3],
                "label": label,
                "iou": max_iou,
                **features
            })

    stats.append({
        "split": split,
        "jumlah_gambar": len(image_paths),
        "jumlah_gt": total_gt,
        "jumlah_kandidat": total_candidates
    })

df = pd.DataFrame(rows)
stats_df = pd.DataFrame(stats)

if df.empty:
    raise ValueError("Tidak ada fitur yang berhasil diekstraksi.")

df.to_csv(result_dir / "features.csv", index=False)
stats_df.to_csv(result_dir / "dataset_stats.csv", index=False)

print("\nStatistik dataset:")
print(stats_df.to_string(index=False))

print("\nJumlah kandidat berdasarkan label:")
print(pd.crosstab(df["split"], df["label"]))

print("\nContoh fitur:")
print(df.head().to_string(index=False))

print("\nHasil tersimpan di folder M2/results")
