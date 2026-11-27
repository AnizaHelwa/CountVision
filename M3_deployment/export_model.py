import argparse
import csv
import hashlib
import json
import platform
import shutil
import statistics
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    output = root / "app" / "models"
    folder = args.dataset.resolve() / "P4" / "valid"
    images = sorted(p for p in (folder / "images").iterdir()
                    if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if len(images) != 23:
        raise ValueError(f"Diperlukan 23 foto validasi P4, ditemukan {len(images)}.")
    selected = [images[i] for i in (0, 5, 11, 16, 22)]
    for image in selected:
        if not (folder / "labels" / f"{image.stem}.txt").is_file():
            raise FileNotFoundError(f"Label tidak ditemukan: {image.stem}")
    import cv2
    import torch
    import ultralytics
    sys.path.insert(0, str(output))
    from inference import load_model, predict_bgr
    metadata_path = output / "model_info.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if ultralytics.__version__ != metadata["training_ultralytics_version"]:
        raise RuntimeError("Gunakan environment training dengan ultralytics 8.4.213 agar konsisten.")
    model = load_model()
    if model.names != {0: "grape"}:
        raise ValueError(f"Kelas model tidak sesuai: {model.names}")
    samples = output / "validation_samples"
    samples.mkdir(exist_ok=True)
    first = cv2.imread(str(selected[0]))
    for _ in range(3):
        predict_bgr(model, first, already_p4=True)
    rows, inference_ms, predict_ms = [], [], []
    for source in selected:
        image = cv2.imread(str(source))
        result = None
        for _ in range(5):
            start = time.perf_counter()
            result = predict_bgr(model, image, already_p4=True)
            predict_ms.append((time.perf_counter() - start) * 1000)
            inference_ms.append(float(result.speed["inference"]))
        label = folder / "labels" / f"{source.stem}.txt"
        gt = sum(bool(line.strip()) for line in label.read_text().splitlines())
        shutil.copy2(source, samples / source.name)
        shutil.copy2(label, samples / label.name)
        annotated = samples / f"{source.stem}_prediction.jpg"
        if not cv2.imwrite(str(annotated), result.plot()):
            raise RuntimeError(f"Gagal menyimpan {annotated}")
        rows.append({"filename": source.name, "split": "valid", "preprocessing": "P4",
                     "already_preprocessed": True, "ground_truth": gt,
                     "predicted_count": len(result.boxes),
                     "absolute_count_error": abs(gt - len(result.boxes)),
                     "sha256": hashlib.sha256(source.read_bytes()).hexdigest()})
        print(f"{source.name}: GT={gt}, prediksi={len(result.boxes)}", flush=True)
    with (samples / "predictions.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    benchmark = {"device": "cpu", "processor": platform.processor(),
                 "platform": platform.platform(), "python": platform.python_version(),
                 "ultralytics": ultralytics.__version__, "torch": torch.__version__,
                 "torch_threads": torch.get_num_threads(), "batch_size": 1,
                 "warmup_calls": 3, "measured_calls": 25, "unique_images": 5,
                 "mean_model_inference_ms": statistics.mean(inference_ms),
                 "median_model_inference_ms": statistics.median(inference_ms),
                 "mean_predict_call_ms": statistics.mean(predict_ms),
                 "scope": "Gambar P4 sudah dalam memori; tidak termasuk baca file, preprocessing M1, load model, rendering, atau jaringan.",
                 "inference_ms": inference_ms, "predict_call_ms": predict_ms}
    (output / "cpu_benchmark.json").write_text(json.dumps(benchmark, indent=2), encoding="utf-8")
    metadata["validation_samples_status"] = "complete: validation_samples/predictions.csv"
    metadata["cpu_benchmark_status"] = "complete: cpu_benchmark.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    archive = shutil.make_archive(str(root / "M3_models_lengkap"), "zip", root, "app")
    print(f"Rata-rata inferensi model CPU: {statistics.mean(inference_ms):.2f} ms/foto")
    print(f"Paket untuk teman website: {archive}")


if __name__ == "__main__":
    main()
