"""grape_pipeline.py - pipeline final M1 (P0 + B1) dan ekspor varian dataset untuk M3.

Dipakai dari kode lain (M2/M3):
    import cv2
    from grape_pipeline import detect_grapes
    boxes, mask = detect_grapes(cv2.imread("foto.jpg"))

Ekspor varian dataset YOLO (untuk membandingkan raw vs preprocessed):
    python grape_pipeline.py --ekspor P0 P2 P3
"""
import argparse
import glob
import os
import shutil

import cv2

import counting as ct
import preprocess as pp

FINAL_PARAMS = {"min_dist": 45, "min_peak": 10, "min_area": 1000, "smooth": 5}

# Varian untuk ekspor. P0 = hanya resize sisi terpanjang 1024 (dianggap "raw" untuk M3).
VARIANTS = {
    "P0": {},
    "P1": {"denoise": {"method": "gaussian", "k": 5}},
    "P2": {"clahe": {"clip": 2.0, "tile": 8}},
    "P3": {"gamma": 1.4, "clahe": {"clip": 2.0, "tile": 8},
           "denoise": {"method": "gaussian", "k": 5}},
    "P4": {"gamma": 1.4, "clahe": {"clip": 2.0, "tile": 8},
           "denoise": {"method": "bilateral", "k": 9},
           "sharpen": {"sigma": 1.5, "amount": 0.7}},
}


def detect_grapes(img_bgr, cfg=None, params=None, normalized=False):
    """Deteksi dan hitung anggur. Mengembalikan (boxes, mask).

    boxes : list (x1, y1, x2, y2). Piksel pada gambar yang sudah di-resize (sisi terpanjang 1024),
            atau 0-1 terhadap ukuran gambar jika normalized=True (berlaku juga untuk gambar asli).
    Jumlah anggur = len(boxes).
    """
    cfg = cfg or pp.FINAL_CONFIG
    params = params or FINAL_PARAMS
    proc = pp.preprocess(img_bgr, cfg)
    mask = pp.make_mask(proc, cfg)
    boxes = ct.count_b1(mask, **params)
    if normalized:
        h, w = mask.shape
        boxes = [(x1 / w, y1 / h, x2 / w, y2 / h) for x1, y1, x2, y2 in boxes]
    return boxes, mask


def export_variant(name, src="data/labels/yolo", dst="data/yolo_variants"):
    """Salin dataset YOLO hasil Roboflow dengan gambar yang diproses. Label disalin apa adanya:
    koordinatnya ter-normalisasi dan prosesnya tidak memotong atau menggeser gambar."""
    cfg = pp.get_config(**VARIANTS[name])
    root = f"{dst}/{name}"
    total = 0
    for split in ("train", "valid", "test"):
        os.makedirs(f"{root}/{split}/images", exist_ok=True)
        os.makedirs(f"{root}/{split}/labels", exist_ok=True)
        for p in sorted(glob.glob(f"{src}/{split}/images/*")):
            img = cv2.imread(p)
            if img is None:
                print("GAGAL baca:", p)
                continue
            cv2.imwrite(f"{root}/{split}/images/{os.path.basename(p)}",
                        pp.preprocess(img, cfg), [cv2.IMWRITE_JPEG_QUALITY, 95])
            total += 1
        for lp in glob.glob(f"{src}/{split}/labels/*.txt"):
            shutil.copy2(lp, f"{root}/{split}/labels/")
    with open(f"{root}/data.yaml", "w") as f:
        f.write(f"path: {os.path.abspath(root)}   # ubah sesuai lokasi folder di Colab\n"
                "train: train/images\nval: valid/images\ntest: test/images\n"
                "nc: 1\nnames: ['grape']\n")
    print(f"{name}: {total} gambar -> {root}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ekspor", nargs="+", choices=list(VARIANTS), required=True)
    for n in ap.parse_args().ekspor:
        export_variant(n)
