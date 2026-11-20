"""tahap4_baseline.py - baseline counting B0 vs B1 dengan protokol tuning yang jujur.

  python tahap4_baseline.py          -> tuning di TRAIN, evaluasi di TRAIN dan VAL
  python tahap4_baseline.py --test   -> evaluasi di TEST (sekali saja, di akhir)

Test sengaja TIDAK disentuh pada mode biasa.
"""
import argparse
import copy
import itertools
import json
import os
import sys
import time

import cv2
import numpy as np
import pandas as pd

import counting as ct
import evaluate as ev
import preprocess as pp

OUT = "hasil_baseline"
LOCK = f"{OUT}/TEST_SUDAH_DIJALANKAN.txt"
OBJECTIVE = "F1@0.3"   # kriteria pemilihan konfigurasi di train (lebih kecil lebih baik)

GRIDS = {                          # tahap 1: parameter mask, dinilai lewat B0
    "hsv": {"s_min": [60, 80, 100, 120, 140], "h_lo": [130],
            "open_k": [7, 9, 11], "fill": [400], "min_area": [500, 1000, 2000, 3500]},
    "lab": {"a_min": [140, 148, 156], "b_max": [135, 145, 155],
            "open_k": [5, 7, 9], "fill": [400, 1500, 4000], "min_area": [200, 500]},
}
GRID_SPLIT = {                     # tahap 2: parameter pemisah buah menempel (B1)
    "min_dist": [25, 35, 45, 55],
    "min_peak": [10, 15, 20],
}

def skor(m):           # lebih kecil = lebih baik
    return -m[OBJECTIVE]


# ---------------- data ----------------
def load_table():
    meta = pd.read_csv("metadata.csv")
    meta = meta[meta["status"] == "dipakai"].drop(columns=["split", "domain"], errors="ignore")
    sp = pd.read_csv("data/splits/split.csv")
    return meta.merge(sp[["filename", "split", "domain"]], on="filename")


def load_images(filenames, max_side=1024):
    imgs = {}
    for fn in filenames:
        img = cv2.imread(f"data/raw/{fn}")
        if img is None:
            sys.exit(f"Gagal membaca data/raw/{fn}")
        imgs[fn] = pp.resize_max(img, max_side)
    return imgs


# ---------------- prediksi & evaluasi ----------------
def predict(img, cfg, method, params):
    proc = pp.preprocess(img, cfg)
    mask = pp.make_mask(proc, cfg)
    boxes = ct.METHODS[method](mask, **params)
    h, w = mask.shape
    norm = [(x1 / w, y1 / h, x2 / w, y2 / h) for x1, y1, x2, y2 in boxes]
    return norm, mask


def run(table, imgs, gt, cfg, method, params, keep_pred=False):
    rows, preds = [], {}
    for _, r in table.iterrows():
        fn = r["filename"]
        boxes, mask = predict(imgs[fn], cfg, method, params)
        row = {"filename": fn, "split": r["split"], "domain": r["domain"],
               "background": r["background"], "pencahayaan": r["pencahayaan"]}
        row.update(ev.evaluate_image(boxes, gt[fn]))
        rows.append(row)
        if keep_pred:
            preds[fn] = (boxes, mask)
    df = pd.DataFrame(rows)
    return (df, preds) if keep_pred else df


def make_cfg(base, space, p):
    cfg = copy.deepcopy(base)
    cfg["color_space"] = space
    if space == "hsv":
        cfg["hsv"]["s_min"] = p["s_min"]
        cfg["hsv"]["hue_ranges"] = [(0, 12), (p["h_lo"], 179)]
    else:
        cfg["lab"]["a_min"], cfg["lab"]["b_max"] = p["a_min"], p["b_max"]
    cfg["morph"]["open_k"] = p["open_k"]
    cfg["morph"]["fill_hole_area"] = p["fill"]
    return cfg


def grid_search(table, imgs, gt, base_cfg, method, space="hsv", mask_choice=None):
    """Mengembalikan daftar hasil terurut; elemen pertama = terbaik di TRAIN."""
    results = []
    if method == "B0":
        g = GRIDS[space]
        keys = list(g)
        combos = list(itertools.product(*g.values()))
        total = len(combos)
        for i, vals in enumerate(combos, 1):
            p = dict(zip(keys, vals))
            cfg = make_cfg(base_cfg, space, p)
            params = {"min_area": p["min_area"]}
            df = run(table, imgs, gt, cfg, "B0", params)
            m = ev.summarize(df).iloc[0]
            results.append((skor(m), m, cfg, params)) 
            print(f"\r  B0 {i}/{total}", end="", flush=True)
    else:   # B1: mask dan min_area dikunci dari hasil B0, hanya pemisah yang dicari
        cfg0, p0 = mask_choice
        combos = list(itertools.product(GRID_SPLIT["min_dist"], GRID_SPLIT["min_peak"]))
        total = len(combos)
        for i, (d, pk) in enumerate(combos, 1):
            params = {"min_dist": d, "min_peak": pk, "min_area": p0["min_area"], "smooth": 5}
            df = run(table, imgs, gt, cfg0, "B1", params)
            m = ev.summarize(df).iloc[0]
            results.append((skor(m), m, cfg0, params))
            print(f"\r  B1 {i}/{total}", end="", flush=True)
    print()
    results.sort(key=lambda t: (t[0], t[1]["MAE"]))
    return results


# ---------------- visualisasi ----------------
def save_vis(imgs, gt, preds, method, folder):
    os.makedirs(folder, exist_ok=True)
    for fn, (boxes, mask) in preds.items():
        img = imgs[fn].copy()
        h, w = img.shape[:2]
        for x1, y1, x2, y2 in gt[fn]:
            cv2.rectangle(img, (int(x1 * w), int(y1 * h)), (int(x2 * w), int(y2 * h)), (0, 255, 0), 2)
        for x1, y1, x2, y2 in boxes:
            cv2.rectangle(img, (int(x1 * w), int(y1 * h)), (int(x2 * w), int(y2 * h)), (0, 0, 255), 2)
        cv2.putText(img, f"GT {len(gt[fn])} | {method} {len(boxes)}", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 255), 3)
        panel = np.hstack([img, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)])
        cv2.imwrite(f"{folder}/{method}_{fn}", panel, [cv2.IMWRITE_JPEG_QUALITY, 80])


def show(title, df):
    print(f"\n=== {title} ===")
    print(ev.summarize(df).round(3).to_string(index=False))


# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true", help="evaluasi di test (sekali saja)")
    ap.add_argument("--force", action="store_true", help="abaikan kunci test (tidak disarankan)")
    ap.add_argument("--space", choices=["hsv", "lab"], default="hsv", help="ruang warna segmentasi")
    args = ap.parse_args()
    global OUT, LOCK
    OUT = f"hasil_v2_{args.space}"
    LOCK = f"{OUT}/TEST_SUDAH_DIJALANKAN.txt"
    os.makedirs(OUT, exist_ok=True)

    table = load_table()
    gt = ev.load_gt()
    missing = [f for f in table["filename"] if f not in gt]
    if missing:
        sys.exit(f"Label tidak ditemukan untuk: {missing[:5]}")

    # ------------------------------------------------ mode TEST
    if args.test:
        if os.path.exists(LOCK) and not args.force:
            sys.exit(f"Test sudah pernah dijalankan ({LOCK}). Menjalankan ulang membuat test ikut "
                     f"menentukan pilihan konfigurasi. Hapus kunci hanya jika kamu paham risikonya.")
        for m in ("B0", "B1"):
            if not os.path.exists(f"{OUT}/best_{m}.json"):
                sys.exit("Jalankan mode biasa dulu (tanpa --test) untuk menentukan konfigurasi.")
        t = table[table["split"] == "test"]
        imgs = load_images(t["filename"])
        for m in ("B0", "B1"):
            saved = json.load(open(f"{OUT}/best_{m}.json"))
            cfg = pp.get_config(**{k: v for k, v in saved["cfg"].items()})
            df, preds = run(t, imgs, gt, cfg, m, saved["params"], keep_pred=True)
            df.to_csv(f"{OUT}/per_foto_{m}_test.csv", index=False)
            show(f"{m} - TEST (keseluruhan)", df)
            print(f"\n{m} - TEST per domain");     print(ev.summarize(df, "domain").to_string())
            print(f"\n{m} - TEST per background"); print(ev.summarize(df, "background").to_string())
            save_vis(imgs, gt, preds, m, f"{OUT}/vis_test")
        open(LOCK, "w").write(time.strftime("test dijalankan %Y-%m-%d %H:%M:%S\n"))
        return

    # ------------------------------------------------ mode tuning (train) + val
    tr = table[table["split"] == "train"]
    va = table[table["split"] == "val"]
    imgs = load_images(pd.concat([tr, va])["filename"])
    base = pp.get_config()
    best = {}

    print(f"Tuning di TRAIN ({len(tr)} foto). Val ({len(va)} foto) hanya untuk evaluasi.")
    t0 = time.time()
    res0 = grid_search(tr, imgs, gt, base, "B0", args.space)
    best["B0"] = (res0[0][2], res0[0][3])
    res1 = grid_search(tr, imgs, gt, base, "B1", args.space, mask_choice=best["B0"])
    best["B1"] = (res1[0][2], res1[0][3])
    print(f"Selesai tuning dalam {time.time() - t0:.0f} detik")

    for name, res in (("B0", res0), ("B1", res1)):
        print(f"\n5 konfigurasi terbaik {name} (di train, urut {OBJECTIVE}):")
        for score, m, cfg, params in res[:5]:
            mo = cfg["morph"]
            if cfg["color_space"] == "hsv":
                warna = f"s_min={cfg['hsv']['s_min']} h_lo={cfg['hsv']['hue_ranges'][1][0]}"
            else:
                warna = f"a_min={cfg['lab']['a_min']} b_max={cfg['lab']['b_max']}"
            print(f"  {OBJECTIVE}={-score:.2f} MAE={m['MAE']:.2f} | {warna} "
                  f"open_k={mo['open_k']} fill={mo['fill_hole_area']} | {params}")

    for m in ("B0", "B1"):
        cfg, params = best[m]
        json.dump({"cfg": cfg, "params": params}, open(f"{OUT}/best_{m}.json", "w"), indent=2)

        df_tr = run(tr, imgs, gt, cfg, m, params)
        df_va, preds = run(va, imgs, gt, cfg, m, params, keep_pred=True)
        df_tr.to_csv(f"{OUT}/per_foto_{m}_train.csv", index=False)
        df_va.to_csv(f"{OUT}/per_foto_{m}_val.csv", index=False)
        show(f"{m} - TRAIN", df_tr)
        show(f"{m} - VAL", df_va)
        print(f"\n{m} - VAL per domain");     print(ev.summarize(df_va, "domain").to_string())
        print(f"\n{m} - VAL per background"); print(ev.summarize(df_va, "background").to_string())
        save_vis(imgs, gt, preds, m, f"{OUT}/vis_val")

    print(f"\nSelesai. Lihat folder {OUT}/ (CSV per foto, best_*.json, vis_val/).")
    print("Test belum disentuh. Jangan jalankan --test sebelum konfigurasi final (tahap 5) dipilih.")


if __name__ == "__main__":
    main()
