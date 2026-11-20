import os
import pandas as pd
from PIL import Image, ImageOps
from pillow_heif import register_heif_opener

register_heif_opener()

SRC = "foto_baru"
RAW = "data/raw"
df = pd.read_csv("metadata.csv")

# lanjutkan nomor dari yang terakhir
n = int(df["filename"].str.extract(r"(\d+)")[0].astype(int).max())

exts = (".jpg", ".jpeg", ".png", ".heic")
files = sorted(f for f in os.listdir(SRC) if f.lower().endswith(exts))

rows = []
for f in files:
    try:
        img = ImageOps.exif_transpose(Image.open(os.path.join(SRC, f))).convert("RGB")
    except Exception as e:
        print(f"GAGAL: {f} ({e})")
        continue
    n += 1
    new_name = f"grape_{n:03d}.jpg"
    dst = os.path.join(RAW, new_name)
    img.save(dst, "JPEG", quality=95)
    w, h = img.size
    rows.append({
        "filename": new_name, "nama_asli": f,
        "lebar": w, "tinggi": h,
        "ukuran_kb": round(os.path.getsize(dst) / 1024),
        "status": "dipakai",
    })

df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)
df.to_csv("metadata.csv", index=False)
print(f"Ditambahkan {len(rows)} foto, total sekarang {len(df)}")