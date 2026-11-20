"""evaluate.py - MAE counting, Precision, Recall.

Semua kotak dalam koordinat ter-normalisasi 0-1, format (x1, y1, x2, y2),
supaya tidak bergantung pada ukuran gambar.
"""
import glob
import os
import re
import numpy as np
import pandas as pd

IOU_THRS = (0.5, 0.3)   # 0.5 = utama, 0.3 = longgar (pelengkap)


def yolo_to_xyxy(x, y, w, h):
    return (x - w / 2, y - h / 2, x + w / 2, y + h / 2)


def load_gt(label_dir="data/labels/yolo"):
    """Baca semua label YOLO -> {filename.jpg: array Nx4 xyxy ter-normalisasi}."""
    gt = {}
    for path in glob.glob(f"{label_dir}/*/labels/*.txt"):
        m = re.search(r"(grape_\d{3})", os.path.basename(path))
        if not m:
            continue
        boxes = []
        with open(path) as f:
            for line in f:
                p = line.split()
                if len(p) >= 5:
                    boxes.append(yolo_to_xyxy(*map(float, p[1:5])))
        gt[m.group(1) + ".jpg"] = np.array(boxes, dtype=float).reshape(-1, 4)
    return gt


def iou_matrix(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / union, 0)


def count_tp(iou, thr):
    """Pencocokan satu-lawan-satu, greedy dari IoU terbesar."""
    if iou.size == 0:
        return 0
    pairs = np.argwhere(iou >= thr)
    if len(pairs) == 0:
        return 0
    order = np.argsort(-iou[pairs[:, 0], pairs[:, 1]])
    used_p, used_g, tp = set(), set(), 0
    for idx in order:
        p, g = pairs[idx]
        if p in used_p or g in used_g:
            continue
        used_p.add(p)
        used_g.add(g)
        tp += 1
    return tp


def evaluate_image(pred_boxes, gt_boxes):
    pred = np.array(pred_boxes, dtype=float).reshape(-1, 4)
    gt = np.asarray(gt_boxes, dtype=float).reshape(-1, 4)
    iou = iou_matrix(pred, gt)
    row = {"n_gt": len(gt), "n_pred": len(pred), "abs_err": abs(len(pred) - len(gt))}
    for t in IOU_THRS:
        row[f"tp{int(t * 100)}"] = count_tp(iou, t)
    return row


def summarize(df, by=None):
    """Ringkasan metrik. P dan R dihitung gabungan (jumlah TP/FP/FN seluruh foto)."""
    def one(g):
        out = {
            "n_foto": len(g),
            "MAE": g["abs_err"].mean(),
            "MAE_rel": g["abs_err"].sum() / max(g["n_gt"].sum(), 1),
            "bias": (g["n_pred"] - g["n_gt"]).mean(),   # + = kelebihan hitung, - = kekurangan
        }
        for t in IOU_THRS:
            k = f"tp{int(t * 100)}"
            p = g[k].sum() / max(g["n_pred"].sum(), 1)
            r = g[k].sum() / max(g["n_gt"].sum(), 1)
            out[f"P@{t}"] = p
            out[f"R@{t}"] = r
            out[f"F1@{t}"] = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        return pd.Series(out)

    if by is None:
        return one(df).to_frame().T
    try:
        return df.groupby(by).apply(one, include_groups=False).round(3)
    except TypeError:   # pandas lama (< 2.2) tidak punya include_groups
        return df.groupby(by).apply(one).round(3)
