"""tahap6_segmentasi.py - eksplorasi metode segmentasi: Otsu dan K-means (dibandingkan dengan HSV P0).

  python tahap6_segmentasi.py                       -> Otsu dan K-means
  python tahap6_segmentasi.py --metode otsu         -> hanya salah satu

Aturan main:
  - Hilir SAMA dengan sistem final: morfologi -> counting B0 / B1 (hanya cara membuat mask yang diganti).
  - Tuning hanya di TRAIN, kriteria F1@0.3 (sama dengan tahap 4-5). Evaluasi di TRAIN dan VAL.
  - TEST tidak dimuat sama sekali. Ini eksperimen eksploratif.
  - Baris "HSV_P0" memakai konfigurasi final yang sudah dibekukan, tanpa tuning ulang (pembanding).
Butuh file di folder yang sama: preprocess.py, counting.py, evaluate.py, tahap4_baseline.py, grape_pipeline.py.
"""
import argparse
import glob
import itertools
import os
import time

import cv2
import numpy as np
import pandas as pd

import counting as ct
import evaluate as ev
import grape_pipeline as gp
import preprocess as pp
import tahap4_baseline as t4

OUT = "hasil_segmentasi"
OPEN_K = [5, 7, 9]
MIN_AREA = [1000, 2000]
MIN_DIST = [25, 35, 45, 55]
MIN_PEAK = [10, 15, 20]
CLOSE_K, FILL_HOLE = 7, 400            # sama dengan P0

OTSU_CHANNELS = ["S", "V", "A", "GRAY"]
KM_SPACES = ["lab", "hsv_cone"]
KM_K = [3, 4, 5]
KM_MAXDIST = [40, 80, 1e9]             # 1e9 = tanpa gerbang jarak
N_SAMPLE = 15000


# ======================= Otsu =======================
def otsu_mask(img, channel):
    if channel in ("S", "V"):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        ch = hsv[..., 1] if channel == "S" else hsv[..., 2]
    elif channel == "A":
        ch = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[..., 1]
    else:
        ch = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, m = cv2.threshold(ch, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # anggur: S tinggi, a* tinggi (foreground = di atas ambang); V dan gray: anggur GELAP -> dibalik
    return 255 - m if channel in ("V", "GRAY") else m


# ======================= K-means =======================
def features(img, space):
    if space == "lab":
        return cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)   # kerucut HSV: hue melingkar -> (x, y)
    ang = hsv[..., 0] * (2 * np.pi / 180)
    r = hsv[..., 1] / 255.0 * hsv[..., 2]
    return np.stack([r * np.cos(ang), r * np.sin(ang), hsv[..., 2]], -1)


def cluster_image(img, space, k):
    f = features(img, space)
    h, w = f.shape[:2]
    flat = f.reshape(-1, 3)
    idx = np.random.default_rng(0).choice(len(flat), min(N_SAMPLE, len(flat)), replace=False)
    cv2.setRNGSeed(0)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.5)
    _, _, centers = cv2.kmeans(flat[idx], k, None, crit, 2, cv2.KMEANS_PP_CENTERS)
    d = np.stack([((flat - c) ** 2).sum(1) for c in centers], 1)
    return d.argmin(1).astype(np.uint8).reshape(h, w), centers


def grape_reference(tr, imgs, gt, space):
    """Warna acuan anggur = median piksel di dalam elips tiap box GT, HANYA dari train."""
    pix = []
    for fn in tr["filename"]:
        img = imgs[fn]
        h, w = img.shape[:2]
        gm = np.zeros((h, w), np.uint8)
        for x1, y1, x2, y2 in gt[fn]:
            c = (int((x1 + x2) / 2 * w), int((y1 + y2) / 2 * h))
            ax = (max(int((x2 - x1) * w / 2 * 0.8), 1), max(int((y2 - y1) * h / 2 * 0.8), 1))
            cv2.ellipse(gm, c, ax, 0, 0, 360, 255, -1)
        p = features(img, space)[gm > 0]
        if len(p):
            pix.append(p[np.random.default_rng(1).choice(len(p), min(1000, len(p)), replace=False)])
    return np.median(np.concatenate(pix), axis=0)


class KMeansCache:
    """Menyimpan hasil klaster untuk SATU pasangan (space, k) sekali waktu (hemat memori)."""
    def __init__(self, imgs, fns):
        self.imgs, self.fns, self.key, self.data = imgs, list(fns), None, {}

    def get(self, space, k):
        if self.key != (space, k):
            self.data = {fn: cluster_image(self.imgs[fn], space, k) for fn in self.fns}
            self.key = (space, k)
        return self.data


def kmeans_choices(cache, refs):
    choices = {}
    for space, k, md in itertools.product(KM_SPACES, KM_K, KM_MAXDIST):
        def mask_of(fn, space=space, k=k, md=md):
            labels, centers = cache.get(space, k)[fn]
            d = np.linalg.norm(centers - refs[space], axis=1)
            c = int(d.argmin())
            if d[c] > md:
                return np.zeros(labels.shape, np.uint8)
            return (labels == c).astype(np.uint8) * 255
        choices[f"{space}_k{k}_maxd{'inf' if md > 1e8 else int(md)}"] = mask_of
    return choices


# ======================= evaluasi & tuning =======================
def cleaned(mask_of, table, open_k):
    morph = {"open_k": open_k, "close_k": CLOSE_K, "fill_hole_area": FILL_HOLE}
    return {fn: pp.clean_mask(mask_of(fn), morph) for fn in table["filename"]}


def evaluate_masks(table, masks, gt, method, params, keep=False):
    rows, preds = [], {}
    for _, r in table.iterrows():
        fn = r["filename"]
        mask = masks[fn]
        boxes = ct.METHODS[method](mask, **params)
        h, w = mask.shape
        norm = [(x1 / w, y1 / h, x2 / w, y2 / h) for x1, y1, x2, y2 in boxes]
        row = {"filename": fn, "split": r["split"], "domain": r["domain"],
               "background": r["background"], "pencahayaan": r["pencahayaan"]}
        row.update(ev.evaluate_image(norm, gt[fn]))
        rows.append(row)
        if keep:
            preds[fn] = (norm, mask)
    df = pd.DataFrame(rows)
    return (df, preds) if keep else df


def tune(name, choices, tr, gt):
    cand, total, i = [], len(choices) * len(OPEN_K), 0
    for label, mask_of in choices.items():
        for ok in OPEN_K:
            masks = cleaned(mask_of, tr, ok)
            for ma in MIN_AREA:
                m = ev.summarize(evaluate_masks(tr, masks, gt, "B0", {"min_area": ma})).iloc[0]
                cand.append((-m["F1@0.3"], m["MAE"], label, ok, ma))
            i += 1
            print(f"\r  {name} B0 {i}/{total}", end="", flush=True)
    print()
    cand.sort(key=lambda t: (t[0], t[1]))
    _, _, label, ok, ma = cand[0]
    masks = cleaned(choices[label], tr, ok)
    res = []
    for d, pk in itertools.product(MIN_DIST, MIN_PEAK):
        params = {"min_dist": d, "min_peak": pk, "min_area": ma, "smooth": 5}
        m = ev.summarize(evaluate_masks(tr, masks, gt, "B1", params)).iloc[0]
        res.append((-m["F1@0.3"], m["MAE"], params))
    res.sort(key=lambda t: (t[0], t[1]))
    print(f"  {name} B0: 5 terbaik di train (F1@0.3, MAE, pilihan mask):")
    for f1, mae, lab, ok_, ma_ in cand[:5]:
        print(f"    F1@0.3={-f1:.3f} MAE={mae:.2f} | {lab} open_k={ok_} min_area={ma_}")
    return {"raw": label, "open_k": ok, "B0": {"min_area": ma}, "B1": res[0][2]}


def rows_for(name, method, split, df):
    parts = []
    a = ev.summarize(df)
    a["scope"], a["group"] = "all", "all"
    parts.append(a)
    for scope in ("domain", "pencahayaan", "background"):
        b = ev.summarize(df, scope).reset_index().rename(columns={scope: "group"})
        b["scope"] = scope
        parts.append(b)
    out = pd.concat(parts, ignore_index=True)
    out.insert(0, "metode", name)
    out.insert(1, "method", method)
    out.insert(2, "split", split)
    return out


def finalize(name, choices, best, tr, va, gt):
    rows = []
    for method in ("B0", "B1"):
        for split, sub in (("train", tr), ("val", va)):
            masks = cleaned(choices[best["raw"]], sub, best["open_k"])
            df, preds = evaluate_masks(sub, masks, gt, method, best[method], keep=True)
            df.to_csv(f"{OUT}/per_foto_{name}_{method}_{split}.csv", index=False)
            rows.append(rows_for(name, method, split, df))
            if split == "val" and method == "B1":
                t4.save_vis(IMGS, gt, preds, "B1", f"{OUT}/vis_val/{name}")
    pd.concat(rows).to_csv(f"{OUT}/ringkasan_{name}.csv", index=False)
    print(f"  {name} terpilih: {best['raw']} open_k={best['open_k']} B0={best['B0']} B1={best['B1']}")


def tampilkan():
    files = sorted(glob.glob(f"{OUT}/ringkasan_*.csv"))
    if not files:
        return
    R = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    cols = ["metode", "method", "MAE", "MAE_rel", "bias", "F1@0.5", "F1@0.3"]
    for split in ("val", "train"):
        d = R[(R["split"] == split) & (R["scope"] == "all")].sort_values(["method", "metode"])
        print(f"\n===== PERBANDINGAN METODE SEGMENTASI ({split.upper()}) =====")
        print(d[cols].round(3).to_string(index=False))
    for scope in ("domain", "pencahayaan", "background"):
        d = R[(R["split"] == "val") & (R["method"] == "B1") & (R["scope"] == scope)]
        for metric in ("MAE", "F1@0.3"):
            print(f"\n----- B1, VAL, {metric} per {scope} -----")
            print(d.pivot(index="metode", columns="group", values=metric).round(2).to_string())


IMGS = {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metode", nargs="+", default=["otsu", "kmeans"], choices=["otsu", "kmeans"])
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    table = t4.load_table()
    table = table[table["split"].isin(["train", "val"])]           # test tidak dimuat
    tr, va = table[table["split"] == "train"], table[table["split"] == "val"]
    gt = ev.load_gt()
    IMGS.update(t4.load_images(table["filename"]))
    t0 = time.time()

    # --- pembanding: HSV P0 yang dibekukan (tanpa tuning ulang) ---
    hsv_cfg = pp.FINAL_CONFIG["hsv"]
    choices = {"hsv_final": lambda fn: pp.segment_hsv(IMGS[fn], hsv_cfg)}
    best = {"raw": "hsv_final", "open_k": pp.FINAL_CONFIG["morph"]["open_k"],
            "B0": {"min_area": gp.FINAL_PARAMS["min_area"]}, "B1": dict(gp.FINAL_PARAMS)}
    print("\n##### HSV_P0 (dibekukan, tanpa tuning)")
    finalize("HSV_P0", choices, best, tr, va, gt)

    if "otsu" in args.metode:
        print("\n##### Otsu")
        choices = {f"otsu_{c}": (lambda fn, c=c: otsu_mask(IMGS[fn], c)) for c in OTSU_CHANNELS}
        best = tune("Otsu", choices, tr, gt)
        finalize("Otsu", choices, best, tr, va, gt)

    if "kmeans" in args.metode:
        print("\n##### K-means")
        cache = KMeansCache(IMGS, table["filename"])
        refs = {s: grape_reference(tr, IMGS, gt, s) for s in KM_SPACES}
        print("  warna acuan anggur (median, dari train):",
              {s: np.round(v, 1).tolist() for s, v in refs.items()})
        choices = kmeans_choices(cache, refs)
        best = tune("Kmeans", choices, tr, gt)
        finalize("Kmeans", choices, best, tr, va, gt)

    tampilkan()
    print(f"\nSelesai dalam {time.time() - t0:.0f} detik. Detail di {OUT}/ (vis_val/<metode>/).")
    print("Test tidak dimuat. Ini eksperimen eksploratif di val.")


if __name__ == "__main__":
    main()
