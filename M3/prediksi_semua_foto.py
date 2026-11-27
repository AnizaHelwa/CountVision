from pathlib import Path

import cv2
from ultralytics import YOLO

from evaluate_yolo import read_labels, evaluate, save_csv


ROOT = Path(__file__).resolve().parent
DATASET = ROOT.parent / "yolo_variants"
OUTPUT = ROOT / "hasil_semua_foto"

CONFIDENCE = {
    "P0": 0.50,
    "P1": 0.60,
    "P2": 0.50,
    "P3": 0.30,
    "P4": 0.50,
}

SPLITS = {
    "train": 61,
    "valid": 23,
    "test": 24,
}

EXTENSIONS = {".jpg", ".jpeg", ".png"}


def cek_input():
    daftar = {}

    for variant in CONFIDENCE:
        weights = ROOT / "models" / f"{variant}_best.pt"

        if not weights.is_file():
            raise FileNotFoundError(
                f"Model tidak ditemukan: {weights}"
            )

        for split, expected in SPLITS.items():
            folder = DATASET / variant / split
            image_folder = folder / "images"

            if not image_folder.is_dir():
                raise FileNotFoundError(
                    f"Folder gambar tidak ditemukan: {image_folder}"
                )

            images = sorted(
                path
                for path in image_folder.iterdir()
                if path.is_file()
                and path.suffix.lower() in EXTENSIONS
            )

            if len(images) != expected:
                raise ValueError(
                    f"{variant}/{split}: ditemukan {len(images)} foto, "
                    f"seharusnya {expected}."
                )

            for image in images:
                label = folder / "labels" / f"{image.stem}.txt"

                if not label.is_file():
                    raise FileNotFoundError(
                        f"Label tidak ditemukan: {label}"
                    )

            daftar[(variant, split)] = images

    return daftar


def simpan_gambar(
    result, labels, detail, variant, split, destination
):
    ground_truth = result.orig_img.copy()
    prediction = result.plot(labels=False, line_width=2)

    height, width = ground_truth.shape[:2]

    for box in labels:
        x1, y1, x2, y2 = [
            int(round(value)) for value in box
        ]

        x1 = max(0, min(width - 1, x1))
        x2 = max(0, min(width - 1, x2))
        y1 = max(0, min(height - 1, y1))
        y2 = max(0, min(height - 1, y2))

        cv2.rectangle(
            ground_truth,
            (x1, y1),
            (x2, y2),
            (0, 200, 0),
            2,
        )

    panels = cv2.hconcat([ground_truth, prediction])

    canvas = cv2.copyMakeBorder(
        panels,
        100, 0, 0, 0,
        cv2.BORDER_CONSTANT,
        value=(25, 25, 25),
    )

    title = (
        f"{variant} | {split} | "
        f"GT: {detail['ground_truth']} | "
        f"Prediksi: {detail['prediksi']} | "
        f"Error: {detail['error']}"
    )

    metrics = (
        f"TP: {detail['TP']} | "
        f"FP: {detail['FP']} | "
        f"FN: {detail['FN']} | "
        "Kiri: anotasi | Kanan: prediksi"
    )

    font = cv2.FONT_HERSHEY_SIMPLEX

    for text, y in [(title, 35), (metrics, 75)]:
        text_width = cv2.getTextSize(
            text, font, 0.7, 2
        )[0][0]

        scale = min(
            0.7,
            0.7 * (canvas.shape[1] - 24)
            / max(text_width, 1),
        )

        cv2.putText(
            canvas,
            text,
            (12, y),
            font,
            scale,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    if not cv2.imwrite(str(destination), canvas):
        raise RuntimeError(
            f"Gagal menyimpan gambar: {destination}"
        )


def proses_split(
    model, variant, split, images, confidence
):
    folder_output = OUTPUT / variant / split
    folder_output.mkdir(parents=True, exist_ok=True)

    predictions = []

    for index, image in enumerate(images, start=1):
        result = model.predict(
            source=str(image),
            conf=0.01,
            iou=0.70,
            imgsz=640,
            device="cpu",
            max_det=300,
            verbose=False,
        )[0]

        height, width = result.orig_shape

        label_file = (
            DATASET / variant / split / "labels"
            / f"{image.stem}.txt"
        )

        labels = read_labels(
            label_file, width, height
        )

        boxes = result.boxes.xyxy.tolist()
        scores = result.boxes.conf.tolist()

        record = (
            image.name,
            labels,
            boxes,
            scores,
        )
        predictions.append(record)

        _, rows = evaluate([record], confidence)
        detail = rows[0]

        keep = result.boxes.conf >= confidence
        result.boxes = result.boxes[keep]

        destination = (
            folder_output / f"{image.name}_hasil.jpg"
        )

        simpan_gambar(
            result,
            labels,
            detail,
            variant,
            split,
            destination,
        )

        print(
            f"[{variant}/{split}] "
            f"{index}/{len(images)} | "
            f"GT={detail['ground_truth']} | "
            f"Prediksi={detail['prediksi']} | "
            f"Error={detail['error']}",
            flush=True,
        )

    summary, details = evaluate(
        predictions, confidence
    )

    save_csv(
        folder_output / "hasil_per_foto.csv",
        details,
    )

    tp = summary["TP"]
    fp = summary["FP"]
    fn = summary["FN"]

    return {
        "Model": f"YOLO {variant}",
        "Split": split,
        "Jumlah Foto": len(images),
        "Jumlah Anotasi": sum(
            row["ground_truth"] for row in details
        ),
        "confidence": confidence,
        "Precision": (
            tp / (tp + fp) if tp + fp else 0.0
        ),
        "Recall": (
            tp / (tp + fn) if tp + fn else 0.0
        ),
        "F1-score": summary["F1-score"],
        "MAE Counting": summary["MAE Counting"],
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "Foto Hitungan Tepat": sum(
            row["error"] == 0 for row in details
        ),
        "Foto Tanpa FP FN": sum(
            row["FP"] == 0 and row["FN"] == 0
            for row in details
        ),
    }


def main():
    print(
        "Memeriksa model, dataset, dan label...",
        flush=True,
    )

    daftar = cek_input()
    OUTPUT.mkdir(parents=True, exist_ok=True)

    summaries = []

    for variant, confidence in CONFIDENCE.items():
        weights = (
            ROOT / "models" / f"{variant}_best.pt"
        )
        model = YOLO(str(weights))

        for split in SPLITS:
            summary = proses_split(
                model,
                variant,
                split,
                daftar[(variant, split)],
                confidence,
            )

            summaries.append(summary)

            save_csv(
                OUTPUT / "ringkasan_per_split.csv",
                summaries,
            )

            print(
                f"\nSelesai {variant}/{split} | "
                f"F1={summary['F1-score']:.2%} | "
                f"MAE={summary['MAE Counting']:.3f}\n",
                flush=True,
            )

    total = sum(
        row["Jumlah Foto"] for row in summaries
    )

    print(
        f"\nSELESAI: {total} gambar hasil "
        "deteksi disimpan."
    )
    print(f"Lokasi hasil: {OUTPUT}")
    print("Ringkasan: ringkasan_per_split.csv")
    print(
        "Hasil train, validasi, dan test "
        "dilaporkan terpisah."
    )


if __name__ == "__main__":
    main()