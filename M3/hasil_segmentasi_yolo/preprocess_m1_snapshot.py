"""preprocess.py - preprocessing dan segmentasi anggur.

Dipakai oleh M1 (eksperimen), M2 (fitur) dan M3 (perbandingan raw vs preprocessed).
Semua langkah dikendalikan satu dict konfigurasi, jadi berpindah dari P0 ke P3
cukup dengan mengubah config, bukan menulis ulang kode.
"""
import copy
import cv2
import numpy as np

# Konfigurasi dasar (P0): hanya resize + segmentasi HSV + morfologi minimal.
# Nilai HSV adalah titik awal dari EDA (train), BUKAN nilai final.
BASE_CONFIG = {
    "max_side": 1024,
    "color_space": "hsv",
    "gamma": None,       # contoh: 0.7 (terang) atau 1.5 (gelap)
    "clahe": None,       # contoh: {"clip": 2.0, "tile": 8}
    "denoise": None,     # contoh: {"method": "gaussian", "k": 5}  (gaussian|median|bilateral)
    "sharpen": None,     # contoh: {"sigma": 1.5, "amount": 0.7}
    "hsv": {
        "hue_ranges": [(0, 12), (130, 179)],   # merah melingkar -> dua rentang
        "s_min": 70,
        "v_min": 15,
        "v_max": 255,
    },
    "lab": {"a_min": 145, "b_max": 145, "l_max": 200},   # titik awal, di-tuning di train
    "morph": {"open_k": 5, "close_k": 7, "fill_hole_area": 400},
}


def get_config(**overrides):
    """Salinan BASE_CONFIG. Contoh: get_config(clahe={"clip": 2.0, "tile": 8})."""
    cfg = copy.deepcopy(BASE_CONFIG)
    for k, v in overrides.items():
        if isinstance(v, dict) and isinstance(cfg.get(k), dict):
            cfg[k].update(v)
        else:
            cfg[k] = v
    return cfg


# ---------------- langkah-langkah dasar ----------------
def resize_max(img, max_side=1024):
    h, w = img.shape[:2]
    s = max_side / max(h, w)
    if s >= 1:
        return img
    return cv2.resize(img, (int(round(w * s)), int(round(h * s))), interpolation=cv2.INTER_AREA)


def apply_gamma(img, gamma):
    table = ((np.arange(256) / 255.0) ** (1.0 / gamma) * 255).astype(np.uint8)
    return cv2.LUT(img, table)


def apply_clahe(img, clip=2.0, tile=8):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile)).apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


def denoise(img, method="gaussian", k=5):
    k = int(k) | 1  # harus ganjil
    if method == "gaussian":
        return cv2.GaussianBlur(img, (k, k), 0)
    if method == "median":
        return cv2.medianBlur(img, k)
    if method == "bilateral":
        return cv2.bilateralFilter(img, k, 50, 50)
    raise ValueError(f"method denoise tidak dikenal: {method}")


def sharpen(img, sigma=1.5, amount=0.7):
    blur = cv2.GaussianBlur(img, (0, 0), sigma)
    return cv2.addWeighted(img, 1 + amount, blur, -amount, 0)


def preprocess(img_bgr, cfg):
    """Resize -> gamma -> CLAHE -> denoise -> sharpen (tiap langkah opsional)."""
    img = resize_max(img_bgr, cfg.get("max_side", 1024))
    if cfg.get("gamma"):
        img = apply_gamma(img, cfg["gamma"])
    if cfg.get("clahe"):
        img = apply_clahe(img, **cfg["clahe"])
    if cfg.get("denoise"):
        img = denoise(img, **cfg["denoise"])
    if cfg.get("sharpen"):
        img = sharpen(img, **cfg["sharpen"])
    return img


# ---------------- segmentasi ----------------
def segment_hsv(img_bgr, hsv_cfg):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mask = np.zeros(hsv.shape[:2], np.uint8)
    for lo, hi in hsv_cfg["hue_ranges"]:
        lower = np.array([lo, hsv_cfg["s_min"], hsv_cfg["v_min"]], np.uint8)
        upper = np.array([hi, 255, hsv_cfg["v_max"]], np.uint8)
        mask |= cv2.inRange(hsv, lower, upper)
    return mask

def segment_lab(img_bgr, lab_cfg):
    L, a, b = cv2.split(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB))   # a,b di OpenCV: 128 = netral
    m = (a >= lab_cfg["a_min"]) & (b <= lab_cfg["b_max"]) & (L <= lab_cfg["l_max"])
    return m.astype(np.uint8) * 255

def fill_small_holes(mask, max_area):
    """Tutup lubang kecil di dalam objek (biasanya highlight/pantulan di kulit anggur)."""
    if max_area <= 0:
        return mask
    inv = (mask == 0).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=4)
    small = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] < max_area]
    if small:
        mask = mask.copy()
        mask[np.isin(lab, small)] = 255
    return mask


def clean_mask(mask, morph_cfg):
    ok, ck = int(morph_cfg["open_k"]), int(morph_cfg["close_k"])
    if ok > 1:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ok, ok)))
    if ck > 1:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ck, ck)))
    return fill_small_holes(mask, morph_cfg.get("fill_hole_area", 0))


def make_mask(img_proc_bgr, cfg):
    """Gambar yang sudah dipreprocess -> mask biner (0/255)."""
    if cfg.get("color_space", "hsv") == "lab":
        raw = segment_lab(img_proc_bgr, cfg["lab"])
    else:
        raw = segment_hsv(img_proc_bgr, cfg["hsv"])
    return clean_mask(raw, cfg["morph"])


# Konfigurasi final bagian M1 (P0 + morfologi), hasil tuning di TRAIN. Parameter B1 ada di grape_pipeline.py.
FINAL_CONFIG = get_config(
    hsv={"s_min": 120, "hue_ranges": [(0, 12), (130, 179)], "v_min": 15, "v_max": 255},
    morph={"open_k": 9, "close_k": 7, "fill_hole_area": 400},
)
