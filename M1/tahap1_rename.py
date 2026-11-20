import os
import pandas as pd
from PIL import Image, ImageOps
from pillow_heif import register_heif_opener

register_heif_opener()   # supaya PIL bisa membaca .heic

SRC = "foto_asli"
RAW = "data/raw"
os.makedirs(RAW, exist_ok=True)

exts = (".jpg", ".jpeg", ".png", ".heic")
files = sorted(f for f in os.listdir(SRC) if f.lower().endswith(exts))
print(f"Ditemukan {len(files)} file")

rows, gagal = [], []
n = 0
for f in files:
    try:
        img = Image.open(os.path.join(SRC, f))
        img = ImageOps.exif_transpose(img).convert("RGB")
    except Exception as e:
        gagal.append((f, str(e)))
        continue

    n += 1
    new_name = f"grape_{n:03d}.jpg"
    dst = os.path.join(RAW, new_name)
    img.save(dst, "JPEG", quality=95)

    w, h = img.size
    rows.append({
        "filename": new_name,
        "nama_asli": f,
        "lebar": w, "tinggi": h,
        "ukuran_kb": round(os.path.getsize(dst) / 1024),
        "background": "", "pencahayaan": "", "posisi": "",
        "jarak": "", "jumlah_gt": "",
    })

df = pd.DataFrame(rows)
df.to_csv("metadata.csv", index=False)
print(df[["filename", "lebar", "tinggi", "ukuran_kb"]].describe())

if gagal:
    print("\nFile yang gagal dibaca:")
    for f, e in gagal:
        print(f" - {f}: {e}")