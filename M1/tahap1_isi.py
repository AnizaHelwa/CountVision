import re, shutil
from datetime import datetime
import pandas as pd

stamp = datetime.now().strftime("%H%M%S")
shutil.copy("metadata.csv", f"metadata_backup_{stamp}.csv")

FIELDS = ["scene_id", "background", "pencahayaan", "posisi",
          "sudut", "jarak", "sumber", "catatan"]

def parse_foto(spec):
    nums = []
    for part in re.split(r"[,\s]+", str(spec).strip()):
        if not part:
            continue
        if ".." in part:
            a, b = part.split("..")
            nums += list(range(int(a), int(b) + 1))
        else:
            nums.append(int(part))
    return nums

meta = pd.read_csv("metadata.csv", sep=None, engine="python", dtype=str)
tabel = pd.read_csv("scene_table.csv", sep=None, engine="python", dtype=str)
tabel = tabel.dropna(how="all")

meta["status"] = meta["status"].fillna("dipakai")
for c in FIELDS:
    if c not in meta.columns:
        meta[c] = pd.NA

idx = dict(zip(meta["filename"], meta.index))
sudah = {}

for r_no, row in tabel.iterrows():
    if pd.isna(row.get("foto")):
        continue
    baris_excel = r_no + 2
    for n in parse_foto(row["foto"]):
        fn = f"grape_{n:03d}.jpg"
        if fn not in idx:
            print(f"[baris {baris_excel}] foto {n} tidak ada di metadata, dilewati")
            continue
        i = idx[fn]
        if meta.at[i, "status"] == "excluded":
            print(f"[baris {baris_excel}] foto {n} berstatus excluded, dilewati")
            continue
        if fn in sudah:
            print(f"[baris {baris_excel}] PERINGATAN: foto {n} juga ada di baris {sudah[fn]}, "
                  f"isian baris ini menimpa yang lama")
        sudah[fn] = baris_excel
        for c in FIELDS:
            val = row.get(c)
            if c == "scene_id" and (pd.isna(val) or str(val).strip() == ""):
                val = f"S{n:03d}"
            if pd.notna(val) and str(val).strip() != "":
                meta.at[i, c] = str(val).strip()

meta.to_csv("metadata.csv", index=False)

pakai = meta[meta["status"] == "dipakai"]
print(f"\nFoto terisi lewat tabel: {len(sudah)} dari {len(pakai)} foto dipakai")
belum = pakai[pakai["scene_id"].isna()]["filename"].tolist()
if belum:
    print("Foto yang belum masuk tabel:", belum)
for c in ["background", "pencahayaan", "posisi", "sudut", "jarak"]:
    print(f"Kolom {c} masih kosong: {pakai[c].isna().sum()} foto")