"""Mode 1: pengolahan citra (milik M1). Mask -> morfologi -> penghitungan B0/B1."""
import cv2
import numpy as np

import counting as ct          # dari folder M1/
import preprocess as pp        # dari folder M1/
from grape_pipeline import FINAL_PARAMS

from .common import Hasil

# warna acuan anggur untuk K-means (median piksel anggur di TRAIN, ruang LAB)
KMEANS_REF = np.array([38.0, 146.0, 135.0], dtype=np.float32)
KMEANS_K, KMEANS_MAXD, N_SAMPLE = 4, 80, 15000


def _mask_hsv(img):
    return pp.segment_hsv(img, pp.FINAL_CONFIG["hsv"])


def _mask_otsu(img):
    s = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[..., 1]
    return cv2.threshold(s, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]


def _mask_kmeans(img):
    f = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    h, w = f.shape[:2]
    flat = f.reshape(-1, 3)
    idx = np.random.default_rng(0).choice(len(flat), min(N_SAMPLE, len(flat)), replace=False)
    cv2.setRNGSeed(0)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.5)
    _, _, centers = cv2.kmeans(flat[idx], KMEANS_K, None, crit, 2, cv2.KMEANS_PP_CENTERS)
    d = np.linalg.norm(centers - KMEANS_REF, axis=1)
    c = int(d.argmin())
    if d[c] > KMEANS_MAXD:
        return np.zeros((h, w), np.uint8)
    lab = np.stack([((flat - ce) ** 2).sum(1) for ce in centers], 1).argmin(1).reshape(h, w)
    return (lab == c).astype(np.uint8) * 255


# parameter = hasil tuning di train (lihat M1/tahap4_baseline.py dan tahap6_segmentasi.py)
METODE = {
    "HSV (final)": dict(mask=_mask_hsv, open_k=9, b0={"min_area": 1000}, b1=dict(FINAL_PARAMS)),
    "Otsu (kanal S)": dict(mask=_mask_otsu, open_k=9, b0={"min_area": 1000},
                           b1={"min_dist": 25, "min_peak": 10, "min_area": 1000, "smooth": 5}),
    "K-means (LAB, k=4)": dict(mask=_mask_kmeans, open_k=5, b0={"min_area": 1000},
                               b1={"min_dist": 45, "min_peak": 10, "min_area": 1000, "smooth": 5}),
}


def predict(img_bgr, metode="HSV (final)", penghitungan="B1"):
    cfg = METODE[metode]
    img = pp.resize_max(img_bgr, 1024)
    morph = {"open_k": cfg["open_k"], "close_k": 7, "fill_hole_area": 400}
    mask = pp.clean_mask(cfg["mask"](img), morph)
    boxes = ct.METHODS[penghitungan](mask, **cfg["b1" if penghitungan == "B1" else "b0"])
    h, w = mask.shape
    norm = [(x1 / w, y1 / h, x2 / w, y2 / h) for x1, y1, x2, y2 in boxes]
    return Hasil(img=img, boxes=norm, mask=mask, boxes_px=boxes, info={"metode": metode, "penghitungan": penghitungan})


def kandidat(img_bgr):
    """Kandidat kotak (B1, HSV final) untuk mode ML. Mengembalikan (img, boxes_norm)."""
    h = predict(img_bgr, "HSV (final)", "B1")
    return h.img, h.boxes
