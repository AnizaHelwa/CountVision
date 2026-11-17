from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parent.parent
result_dir = root / "M2" / "results"
data_dir = root / "yolo_variants" / "P0" / "test"

df = pd.read_csv(result_dir / "features.csv")
test = df[df["split"] == "test"].copy()

exclude = [
    "filename", "split", "candidate_id",
    "x1", "y1", "x2", "y2", "label", "iou"
]

features = [c for c in df.columns if c not in exclude]

svm = joblib.load(result_dir / "svm_model.pkl")
rf = joblib.load(result_dir / "rf_model.pkl")

test["svm"] = svm.predict(test[features])
test["rf"] = rf.predict(test[features])


def read_labels(path, width, height):
    boxes = []

    if not path.exists():
        raise FileNotFoundError(path)

    for line in path.read_text().splitlines():
        if not line.strip():
            continue

        cls, xc, yc, bw, bh = map(float, line.split())

        if int(cls) != 0:
            continue

        boxes.append((
            (xc - bw / 2) * width,
            (yc - bh / 2) * height,
            (xc + bw / 2) * width,
            (yc + bh / 2) * height
        ))

    return boxes


def calculate_iou(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])

    intersection = max(0, x2-x1) * max(0, y2-y1)

    area_a = max(0, a[2]-a[0]) * max(0, a[3]-a[1])
    area_b = max(0, b[2]-b[0]) * max(0, b[3]-b[1])

    return intersection / (area_a + area_b - intersection + 1e-8)


def match_boxes(pred_boxes, gt_boxes):
    pairs = []

    for i, pred in enumerate(pred_boxes):
        for j, gt in enumerate(gt_boxes):
            score = calculate_iou(pred, gt)

            if score >= 0.5:
                pairs.append((score, i, j))

    pairs.sort(reverse=True)

    used_pred = set()
    used_gt = set()

    for score, i, j in pairs:
        if i not in used_pred and j not in used_gt:
            used_pred.add(i)
            used_gt.add(j)

    tp = len(used_pred)
    fp = len(pred_boxes) - tp
    fn = len(gt_boxes) - tp

    return tp, fp, fn


models = {
    "PCD M1": None,
    "SVM": "svm",
    "Random Forest": "rf"
}

summary = []
details = []

image_paths = sorted(
    p for p in (data_dir / "images").iterdir()
    if p.suffix.lower() in [".jpg", ".jpeg", ".png"]
)

for model_name, column in models.items():
    total_tp = 0
    total_fp = 0
    total_fn = 0
    errors = []

    for path in image_paths:
        img = cv2.imread(str(path))

        if img is None:
            raise ValueError(f"Gambar gagal dibaca: {path.name}")

        h, w = img.shape[:2]

        gt_boxes = read_labels(
            data_dir / "labels" / f"{path.stem}.txt",
            w, h
        )

        rows = test[test["filename"] == path.name]

        if column is not None:
            rows = rows[rows[column] == 1]

        pred_boxes = rows[
            ["x1", "y1", "x2", "y2"]
        ].to_numpy().tolist()

        tp, fp, fn = match_boxes(pred_boxes, gt_boxes)

        gt_count = len(gt_boxes)
        pred_count = len(pred_boxes)
        error = abs(gt_count - pred_count)

        total_tp += tp
        total_fp += fp
        total_fn += fn
        errors.append(error)

        details.append({
            "model": model_name,
            "filename": path.name,
            "ground_truth": gt_count,
            "prediksi": pred_count,
            "error": error,
            "TP": tp,
            "FP": fp,
            "FN": fn
        })

    precision = (
        total_tp / (total_tp + total_fp)
        if total_tp + total_fp else 0
    )

    recall = (
        total_tp / (total_tp + total_fn)
        if total_tp + total_fn else 0
    )

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0
    )

    summary.append({
        "Model": model_name,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1,
        "MAE Counting": np.mean(errors),
        "TP": total_tp,
        "FP": total_fp,
        "FN": total_fn
    })


summary_df = pd.DataFrame(summary)
detail_df = pd.DataFrame(details)

summary_df.to_csv(
    result_dir / "test_evaluation.csv",
    index=False
)

detail_df.to_csv(
    result_dir / "counting_details.csv",
    index=False
)

print("Jumlah gambar test:", len(image_paths))

print("\nHasil evaluasi test:")
print(summary_df.round(3).to_string(index=False))

print("\nContoh hasil penghitungan:")
print(detail_df.head(10).to_string(index=False))

plt.figure(figsize=(8, 5))
plt.bar(
    summary_df["Model"],
    summary_df["MAE Counting"]
)

plt.ylabel("MAE Counting")
plt.title("Perbandingan Kesalahan Penghitungan Anggur")
plt.tight_layout()

plt.savefig(
    result_dir / "mae_comparison.png",
    dpi=150
)
plt.close()

print("\nHasil evaluasi tersimpan di folder M2/results")
