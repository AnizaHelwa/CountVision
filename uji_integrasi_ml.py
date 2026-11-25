"""Uji integrasi mode ML: bandingkan hitungan website dengan prediksi 23 foto val dari M2.

Jalankan dari ROOT repositori:   python uji_integrasi_ml.py
Butuh: M2/inferensi_ml.py, M2/results/*.pkl, M2/results/prediksi_validasi_lengkap.csv, dan foto di
M1/data/raw/ (atau gambar ekspor Roboflow di M1/data/labels/yolo/valid/images/).
"""
import sys
from pathlib import Path

import cv2
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / "M1"), str(ROOT / "app")]
from predictors import ml            # noqa: E402

csv = ROOT / "M2" / "results" / "prediksi_validasi_lengkap.csv"
d = pd.read_csv(csv)
rows = []
for r in d.itertuples():
    kandidat = [ROOT / "M1" / "data" / "labels" / "yolo" / "valid" / "images" / r.filename,   # gambar yang kemungkinan dipakai M2
                ROOT / "M1" / "data" / "raw" / f"{r.id_foto}.jpg"]
    p = next((x for x in kandidat if x.exists()), None)
    img = cv2.imread(str(p)) if p else None
    if img is None:
        print(f"LEWATI {r.id_foto}: gambar tidak ditemukan")
        continue
    s, f = ml.predict(img, "SVM"), ml.predict(img, "Random Forest")
    rows.append(dict(foto=r.id_foto, GT=r.jumlah_gt, kandidat=s.info["kandidat"],
                     web_SVM=s.count, M2_SVM=r.SVM, web_RF=f.count, M2_RF=r.RF,
                     sumber="roboflow" if "labels" in str(p) else "raw"))
t = pd.DataFrame(rows)
if t.empty:
    sys.exit("Tidak ada gambar yang bisa diuji.")
t["beda_SVM"] = t.web_SVM - t.M2_SVM
t["beda_RF"] = t.web_RF - t.M2_RF
print(t.to_string(index=False))
n = len(t)
print(f"\nFoto diuji: {n}")
for m in ("SVM", "RF"):
    web, m2 = t[f"web_{m}"], t[f"M2_{m}"]
    print(f"{m}: identik dengan M2 di {(web == m2).sum()}/{n} foto | selisih mutlak rata-rata "
          f"{(web - m2).abs().mean():.2f} | MAE web {(web - t.GT).abs().mean():.2f} vs MAE M2 {(m2 - t.GT).abs().mean():.2f}")
