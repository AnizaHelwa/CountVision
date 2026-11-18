import csv
from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent
THRESHOLDS = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]


def read_labels(path, width, height):
    boxes = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue

        cls, x, y, w, h = map(float, line.split())
        if cls == 0:
            boxes.append((
                (x - w / 2) * width,
                (y - h / 2) * height,
                (x + w / 2) * width,
                (y + h / 2) * height,
            ))
    return boxes


def iou(a, b):
    overlap_width = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    overlap_height = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    intersection = overlap_width * overlap_height
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return intersection / (area_a + area_b - intersection + 1e-8)


def count_matches(predictions, labels):
    pairs = []
    for pred_index, prediction in enumerate(predictions):
        for label_index, label in enumerate(labels):
            score = iou(prediction, label)
            if score >= 0.5:
                pairs.append((score, pred_index, label_index))

    # Seperti M2, cocokkan kotak satu lawan satu dari IoU tertinggi.
    used_predictions = set()
    used_labels = set()
    for _, pred_index, label_index in sorted(pairs, reverse=True):
        if pred_index not in used_predictions and label_index not in used_labels:
            used_predictions.add(pred_index)
            used_labels.add(label_index)

    return len(used_predictions)


def predict_split(model, variant, split, expected_images):
    folder = ROOT.parent / "yolo_variants" / variant / split
    images = sorted(
        path for path in (folder / "images").iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if len(images) != expected_images:
        raise ValueError(
            f"{folder}: ada {len(images)} foto, seharusnya {expected_images}"
        )

    predictions = []
    for image in images:
        result = model.predict(
            source=str(image),
            conf=0.01,
            iou=0.7,
            imgsz=640,
            device="cpu",
            max_det=300,
            verbose=False,
        )[0]

        height, width = result.orig_shape
        label_file = folder / "labels" / f"{image.stem}.txt"
        labels = read_labels(label_file, width, height)
        boxes = result.boxes.xyxy.tolist()
        scores = result.boxes.conf.tolist()
        predictions.append((image.name, labels, boxes, scores))

    return predictions


def evaluate(predictions, threshold):
    rows = []
    for filename, labels, boxes, scores in predictions:
        kept_boxes = [
            box for box, score in zip(boxes, scores) if score >= threshold
        ]
        actual_count = len(labels)
        predicted_count = len(kept_boxes)
        tp = count_matches(kept_boxes, labels)

        rows.append({
            "filename": filename,
            "ground_truth": actual_count,
            "prediksi": predicted_count,
            "error": abs(predicted_count - actual_count),
            "TP": tp,
            "FP": predicted_count - tp,
            "FN": actual_count - tp,
        })

    tp = sum(row["TP"] for row in rows)
    fp = sum(row["FP"] for row in rows)
    fn = sum(row["FN"] for row in rows)
    denominator = 2 * tp + fp + fn

    summary = {
        "confidence": threshold,
        "F1-score": 2 * tp / denominator if denominator else 0.0,
        "MAE Counting": sum(row["error"] for row in rows) / len(rows),
        "TP": tp,
        "FP": fp,
        "FN": fn,
    }
    return summary, rows


def save_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    output = ROOT / "hasil_evaluasi_yolo"
    output.mkdir(exist_ok=True)
    summaries = []

    for variant in ("P0", "P1", "P2", "P3", "P4"):
        weights = ROOT / "models" / f"{variant}_best.pt"
        if not weights.is_file():
            raise FileNotFoundError(f"Bobot model tidak ditemukan: {weights}")

        model = YOLO(str(weights))

        print(f"Memilih confidence {variant} dari 23 foto validasi...", flush=True)
        validation = predict_split(model, variant, "valid", expected_images=23)
        scores = [evaluate(validation, threshold)[0] for threshold in THRESHOLDS]
        best = max(scores, key=lambda row: (row["F1-score"], -row["MAE Counting"]))
        save_csv(output / f"{variant}_validasi.csv", scores)

        confidence = best["confidence"]
        print(f"Menguji {variant} dengan confidence {confidence:.2f}...", flush=True)
        test = predict_split(model, variant, "test", expected_images=24)
        summary, details = evaluate(test, confidence)
        save_csv(output / f"{variant}_test_per_foto.csv", details)
        summaries.append({"Model": f"YOLO {variant}", **summary})

    save_csv(output / "ringkasan_test.csv", summaries)

    for row in summaries:
        print(
            f"{row['Model']} | Confidence: {row['confidence']:.2f} | "
            f"F1: {row['F1-score']:.2%} | "
            f"MAE: {row['MAE Counting']:.3f} butir/foto"
        )

    print(f"Hasil tersimpan di {output}")


if __name__ == "__main__":
    main()