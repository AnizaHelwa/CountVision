import argparse
import csv
import hashlib
import importlib.util
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
DATA = PROJECT / "yolo_variants" / "P5_SEG"
OUTPUT = ROOT / "hasil_segmentasi_yolo"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def prepare():
    import cv2
    source = PROJECT / "yolo_variants" / "P0"
    module_path = PROJECT / "M1" / "preprocess.py"
    if not module_path.is_file():
        raise FileNotFoundError(f"Tidak ditemukan: {module_path}")
    if DATA.exists():
        raise FileExistsError(f"{DATA} sudah ada. Dataset tidak ditimpa.")
    module = load_module("m1_segmentation", module_path)
    evaluator = load_module("m3_evaluator", ROOT / "evaluate_yolo.py")
    plans = {}
    seen = set()
    for split, expected in (("train", 61), ("valid", 23), ("test", 24)):
        images = sorted(p for p in (source / split / "images").iterdir()
                        if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
        if len(images) != expected:
            raise ValueError(f"{split}: {len(images)} foto, seharusnya {expected}.")
        for path in images:
            if path.stem in seen:
                raise ValueError(f"Nama foto berulang antar split: {path.name}")
            seen.add(path.stem)
            label = source / split / "labels" / f"{path.stem}.txt"
            for line in label.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                values = list(map(float, line.split()))
                if len(values) != 5 or values[0] != 0 or not all(0 <= x <= 1 for x in values[1:]):
                    raise ValueError(f"Label YOLO tidak sesuai: {label}")
        plans[split] = images
    OUTPUT.mkdir(exist_ok=True)
    preview_dir = OUTPUT / "preview"
    preview_dir.mkdir(exist_ok=True)
    rows = []
    for split, images in plans.items():
        for subdir in ("images", "labels", "masks"):
            (DATA / split / subdir).mkdir(parents=True, exist_ok=True)
        for index, path in enumerate(images):
            image = cv2.imread(str(path))
            if image is None:
                raise ValueError(f"Gagal membaca {path}")
            processed = module.preprocess(image, module.FINAL_CONFIG)
            mask = module.make_mask(processed, module.FINAL_CONFIG)
            segmented = cv2.bitwise_and(processed, processed, mask=mask)
            destination = DATA / split / "images" / path.name
            if not cv2.imwrite(str(destination), segmented, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                raise RuntimeError(f"Gagal menyimpan {destination}")
            if not cv2.imwrite(str(DATA / split / "masks" / f"{path.stem}.png"), mask):
                raise RuntimeError("Gagal menyimpan mask")
            label = source / split / "labels" / f"{path.stem}.txt"
            shutil.copy2(label, DATA / split / "labels" / label.name)
            h, w = processed.shape[:2]
            boxes = evaluator.read_labels(label, w, h)
            empty = 0
            for x1, y1, x2, y2 in boxes:
                roi = mask[max(0, int(y1)):min(h, int(y2) + 1), max(0, int(x1)):min(w, int(x2) + 1)]
                empty += int(not roi.any())
            rows.append({"split": split, "filename": path.name, "ground_truth": len(boxes),
                         "mask_fraction": float((mask > 0).mean()), "empty_gt_regions": empty,
                         "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                         "label_sha256": hashlib.sha256(label.read_bytes()).hexdigest()})
            if index < 3:
                preview = cv2.hconcat([processed, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), segmented])
                scale = min(1, 1500 / preview.shape[1])
                preview = cv2.resize(preview, (round(preview.shape[1] * scale), round(preview.shape[0] * scale)))
                cv2.imwrite(str(preview_dir / f"{split}_{path.stem}.jpg"), preview)
        print(f"{split}: {len(images)} gambar selesai", flush=True)
    config = {"path": DATA.as_posix(), "train": "train/images", "val": "valid/images",
              "test": "test/images", "nc": 1, "names": ["grape"]}
    import yaml
    (DATA / "data.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    write_csv(OUTPUT / "dataset_audit.csv", rows)
    shutil.copy2(module_path, OUTPUT / "preprocess_m1_snapshot.py")
    manifest = {"variant": "P5_SEG", "source": "P0", "method": "M1 HSV + morphology mask applied to BGR image",
                "config": module.FINAL_CONFIG, "labels": "Original ground truth retained, including objects removed by mask",
                "preprocess_sha256": hashlib.sha256(module_path.read_bytes()).hexdigest(),
                "note": "Full-frame masked RGB/BGR image, not instance segmentation and not B1 candidate crops"}
    (OUTPUT / "experiment.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Dataset siap: {DATA}")
    print(f"Periksa preview train dan audit: {OUTPUT}")


def train(epochs):
    from ultralytics import YOLO
    if not (DATA / "data.yaml").is_file():
        raise FileNotFoundError("Jalankan tahap prepare terlebih dahulu.")
    initial = PROJECT / "yolo11n.pt"
    model = YOLO(str(initial) if initial.is_file() else "yolo11n.pt")
    model.train(data=str(DATA / "data.yaml"), epochs=epochs, imgsz=640, batch=4,
                device="cpu", workers=0, seed=42, deterministic=True,
                project=str(ROOT / "training_runs"), name=f"P5_SEG_train{epochs}",
                exist_ok=False)
    save_dir = Path(model.trainer.save_dir)
    OUTPUT.mkdir(exist_ok=True)
    (OUTPUT / "latest_training.json").write_text(json.dumps({"save_dir": str(save_dir),
         "best": str(save_dir / "weights" / "best.pt"), "epochs": epochs}, indent=2), encoding="utf-8")
    print(f"Training tersimpan: {save_dir}")


def evaluate_model():
    from ultralytics import YOLO
    evaluator = load_module("m3_evaluator", ROOT / "evaluate_yolo.py")
    training = json.loads((OUTPUT / "latest_training.json").read_text(encoding="utf-8"))
    if training["epochs"] != 30:
        raise ValueError("Evaluasi perbandingan memerlukan training 30 epoch. Jalankan train --epochs 30.")
    weights = Path(training["best"])
    model = YOLO(str(weights))
    validation = evaluator.predict_split(model, "P5_SEG", "valid", 23)
    scores = [evaluator.evaluate(validation, t)[0] for t in evaluator.THRESHOLDS]
    best = max(scores, key=lambda row: (row["F1-score"], -row["MAE Counting"]))
    write_csv(OUTPUT / "validation_thresholds.csv", scores)
    test = evaluator.predict_split(model, "P5_SEG", "test", 24)
    summary, details = evaluator.evaluate(test, best["confidence"])
    write_csv(OUTPUT / "test_per_foto.csv", details)
    tp, fp, fn = (summary[k] for k in ("TP", "FP", "FN"))
    result = {"Model": "YOLO P5_SEG", **summary,
              "Precision": tp / (tp + fp) if tp + fp else 0,
              "Recall": tp / (tp + fn) if tp + fn else 0,
              "Jumlah Foto": 24}
    write_csv(OUTPUT / "ringkasan_test.csv", [result])
    (ROOT / "models").mkdir(exist_ok=True)
    shutil.copy2(weights, ROOT / "models" / "P5_SEG_best.pt")
    original = ROOT / "hasil_evaluasi_yolo" / "ringkasan_test.csv"
    if original.is_file():
        with original.open(encoding="utf-8-sig") as file:
            baseline = list(csv.DictReader(file))
        fields = ["Model", "confidence", "F1-score", "MAE Counting", "TP", "FP", "FN"]
        combined = [{k: row[k] for k in fields} for row in baseline + [result]]
        write_csv(OUTPUT / "perbandingan_varian.csv", combined)
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["prepare", "train", "evaluate"])
    parser.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("epochs minimal 1")
    if args.stage == "prepare":
        prepare()
    elif args.stage == "train":
        train(args.epochs)
    else:
        evaluate_model()


if __name__ == "__main__":
    main()
