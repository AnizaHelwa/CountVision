"""tahap5_pipeline.py - membandingkan pipeline preprocessing P0-P4.

  python tahap5_pipeline.py                      -> jalankan P0..P4
  python tahap5_pipeline.py --pipelines P2 P3    -> hanya sebagian (hasil lama tetap dipakai di tabel)

Aturan main (sama untuk semua pipeline, supaya perbandingannya adil):
  - tiap pipeline di-tuning SENDIRI di TRAIN dengan grid dan kriteria yang sama (tahap4_baseline.py)
  - dievaluasi di TRAIN dan VAL
  - TEST tidak disentuh
Memakai ulang fungsi dari tahap4_baseline.py, jadi file itu harus tetap ada di folder yang sama.
"""
import argparse
import copy
import glob
import json
import os
import time

import pandas as pd

import evaluate as ev
import preprocess as pp
import tahap4_baseline as t4

OUT = "hasil_tahap5"
SPACE = "hsv"

CLAHE = {"clip": 2.0, "tile": 8}
PIPELINES = {
    "P0": {},                                                     # resize + HSV (baseline)
    "P1": {"denoise": {"method": "gaussian", "k": 5}},            # + Gaussian
    "P2": {"clahe": CLAHE},                                       # + CLAHE
    "P3": {"gamma": 1.4, "clahe": CLAHE,                          # + gamma + CLAHE + Gaussian
           "denoise": {"method": "gaussian", "k": 5}},
    "P4": {"gamma": 1.4, "clahe": CLAHE,                          # P3 dengan bilateral + sharpening
           "denoise": {"method": "bilateral", "k": 9},
           "sharpen": {"sigma": 1.5, "amount": 0.7}},
}
STEPS = ["gamma", "clahe", "denoise", "sharpen"]


def rows_for(name, method, split, df):
    parts = []
    a = ev.summarize(df)
    a["scope"], a["group"] = "all", "all"
    parts.append(a)
    for scope in ("domain", "pencahayaan"):
        b = ev.summarize(df, scope).reset_index().rename(columns={scope: "group"})
        b["scope"] = scope
        parts.append(b)
    out = pd.concat(parts, ignore_index=True)
    out.insert(0, "pipeline", name)
    out.insert(1, "method", method)
    out.insert(2, "split", split)
    return out


def run_pipeline(name, table, raw_imgs, gt):
    over = PIPELINES[name]
    tr = table[table["split"] == "train"]
    va = table[table["split"] == "val"]
    base_full = pp.get_config(**over)
    base_mask = pp.get_config()          # tanpa langkah preprocessing (sudah diterapkan di bawah)

    t0 = time.time()
    # preprocessing diterapkan SEKALI untuk train; grid search hanya menyetel mask
    pre = {fn: pp.preprocess(raw_imgs[fn], base_full) for fn in tr["filename"]}
    print(f"\n##### {name}: {over if over else 'tanpa preprocessing tambahan'}")
    res0 = t4.grid_search(tr, pre, gt, base_mask, "B0", SPACE)
    best = {"B0": (res0[0][2], res0[0][3])}
    res1 = t4.grid_search(tr, pre, gt, base_mask, "B1", SPACE, mask_choice=best["B0"])
    best["B1"] = (res1[0][2], res1[0][3])

    rows = []
    for m, (cfg_mask, params) in best.items():
        cfg_full = copy.deepcopy(cfg_mask)
        for k in STEPS:
            cfg_full[k] = copy.deepcopy(base_full[k])
        json.dump({"cfg": cfg_full, "params": params},
                  open(f"{OUT}/best_{name}_{m}.json", "w"), indent=2)
        for split_name, sub in (("train", tr), ("val", va)):
            df, preds = t4.run(sub, raw_imgs, gt, cfg_full, m, params, keep_pred=True)
            df.to_csv(f"{OUT}/per_foto_{name}_{m}_{split_name}.csv", index=False)
            rows.append(rows_for(name, m, split_name, df))
            if split_name == "val":
                t4.save_vis(raw_imgs, gt, preds, m, f"{OUT}/vis_val/{name}")
        c = cfg_mask
        print(f"  {m} terpilih: s_min={c['hsv']['s_min']} open_k={c['morph']['open_k']} {params}")
    pd.concat(rows).to_csv(f"{OUT}/ringkasan_{name}.csv", index=False)
    print(f"  {name} selesai dalam {time.time() - t0:.0f} detik")


def tampilkan():
    files = sorted(glob.glob(f"{OUT}/ringkasan_*.csv"))
    if not files:
        return
    R = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    cols = ["pipeline", "method", "MAE", "MAE_rel", "bias", "F1@0.5", "F1@0.3"]
    for split in ("val", "train"):
        d = R[(R["split"] == split) & (R["scope"] == "all")].sort_values(["method", "pipeline"])
        print(f"\n===== PERBANDINGAN PIPELINE ({split.upper()}) =====")
        print(d[cols].round(3).to_string(index=False))
    for scope in ("pencahayaan", "domain"):
        d = R[(R["split"] == "val") & (R["method"] == "B1") & (R["scope"] == scope)]
        if d.empty:
            continue
        for metric in ("MAE", "F1@0.3"):
            print(f"\n----- B1, VAL, {metric} per {scope} -----")
            print(d.pivot(index="pipeline", columns="group", values=metric).round(3).to_string())
    n = R[(R["split"] == "val") & (R["method"] == "B1") & (R["scope"] == "pencahayaan")]
    print("\nJumlah foto val per pencahayaan:",
          n.drop_duplicates("group").set_index("group")["n_foto"].astype(int).to_dict())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pipelines", nargs="+", default=list(PIPELINES), choices=list(PIPELINES))
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    table = t4.load_table()
    gt = ev.load_gt()
    sub = table[table["split"].isin(["train", "val"])]      # test tidak dimuat sama sekali
    raw_imgs = t4.load_images(sub["filename"])

    for name in args.pipelines:
        run_pipeline(name, sub, raw_imgs, gt)
    tampilkan()
    print(f"\nSelesai. Detail ada di {OUT}/ (ringkasan_*.csv, per_foto_*.csv, vis_val/<pipeline>/).")
    print("Test belum disentuh.")


if __name__ == "__main__":
    main()
