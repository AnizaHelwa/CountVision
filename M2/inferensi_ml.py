from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd
from skimage.feature import local_binary_pattern


folder_model = Path(__file__).resolve().parent / "results"

kolom_fitur = [
    "mean_h", "mean_s", "mean_v",
    "std_h", "std_s", "std_v",
    "area", "perimeter", "circularity",
    "aspect_ratio", "fill_ratio"
] + [f"lbp_{i}" for i in range(10)]


def ekstrak_fitur(img, mask, box):
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
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    area = float(binary.sum())
    perimeter = sum(cv2.arcLength(c, True) for c in contours)

    circularity = 4 * np.pi * area / (perimeter ** 2 + 1e-8)
    aspect_ratio = (x2 - x1) / max(y2 - y1, 1)
    fill_ratio = area / max((x2 - x1) * (y2 - y1), 1)

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, 8, 1, method="uniform")

    values = lbp[binary > 0]
    hist, _ = np.histogram(values, bins=np.arange(11))
    hist = hist / (hist.sum() + 1e-8)

    fitur = {
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

    for i, nilai in enumerate(hist):
        fitur[f"lbp_{i}"] = nilai

    return fitur


def load_model(nama_model):
    if nama_model == "SVM":
        path = folder_model / "svm_model.pkl"
    elif nama_model == "Random Forest":
        path = folder_model / "rf_model.pkl"
    else:
        raise ValueError("Model harus SVM atau Random Forest")

    return joblib.load(path)


def prediksi_ml(img, mask, boxes, model):
    if img.shape[:2] != mask.shape[:2]:
        raise ValueError("Ukuran gambar dan mask harus sama")

    daftar_fitur = []
    kotak_valid = []

    for box in boxes:
        fitur = ekstrak_fitur(img, mask, box)

        if fitur is not None:
            daftar_fitur.append(fitur)
            kotak_valid.append(box)

    if not daftar_fitur:
        return []

    data = pd.DataFrame(daftar_fitur, columns=kolom_fitur)

    prediksi = model.predict(data)

    hasil = [
        box for box, label in zip(kotak_valid, prediksi)
        if int(label) == 1
    ]

    return hasil
