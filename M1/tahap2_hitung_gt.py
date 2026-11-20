import os, re, glob, shutil
import pandas as pd

LABEL_DIR = "data/labels/yolo"   # sesuaikan jika struktur folder beda
SPLIT_DIR = {"train": "train", "valid": "val", "test": "test"}  # nama folder Roboflow -> nama di split.csv

def iou(a, b):
    ax1, ay1, ax2, ay2 = a[0]-a[2]/2, a[1]-a[3]/2, a[0]+a[2]/2, a[1]+a[3]/2
    bx1, by1, bx2, by2 = b[0]-b[2]/2, b[1]-b[3]/2, b[0]+b[2]/2, b[1]+b[3]/2
    iw = max(0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    union = a[2]*a[3] + b[2]*b[3] - inter
    return inter / union if union > 0 else 0

rows, masalah = [], []
for rf_split, split_name in SPLIT_DIR.items():
    for path in glob.glob(f"{LABEL_DIR}/{rf_split}/labels/*.txt"):
        m = re.search(r"(grape_\d{3})", os.path.basename(path))
        if not m:
            continue
        fn = m.group(1) + ".jpg"
        boxes = []
        with open(path) as f:
            for line in f:
                p = line.split()
                if len(p) >= 5:
                    boxes.append(tuple(map(float, p[1:5])))
        rows.append({"filename": fn, "split_roboflow": split_name, "jumlah_gt": len(boxes)})

        if len(boxes) == 0:
            masalah.append((fn, "tidak ada box"))
        for b in boxes:
            if b[2] < 0.01 or b[3] < 0.01:
                masalah.append((fn, f"box sangat kecil {b[2]:.3f}x{b[3]:.3f}"))
            if b[2] * b[3] > 0.25:
                masalah.append((fn, f"box sangat besar (luas {b[2]*b[3]:.2f})"))
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                if iou(boxes[i], boxes[j]) > 0.8:
                    masalah.append((fn, "dua box hampir identik (duplikat?)"))

gt = pd.DataFrame(rows)
dup = gt[gt.duplicated("filename", keep=False)]
if len(dup):
    print("PERINGATAN: foto muncul lebih dari sekali:", dup["filename"].unique().tolist())

# cocokkan dengan split.csv
sp = pd.read_csv("data/splits/split.csv")
cek = sp.merge(gt, on="filename", how="outer", indicator=True)
print("Foto di split.csv tanpa file label:", cek[cek["_merge"] == "left_only"]["filename"].tolist())
salah = cek[(cek["_merge"] == "both") & (cek["split"] != cek["split_roboflow"])]
print("Foto dengan split BERBEDA dari split.csv:", salah["filename"].tolist())

# tulis ke metadata
shutil.copy("metadata.csv", "metadata_backup_gt.csv")
meta = pd.read_csv("metadata.csv")
meta = meta.drop(columns=["jumlah_gt"], errors="ignore").merge(
    gt[["filename", "jumlah_gt"]].drop_duplicates("filename"), on="filename", how="left")
meta.to_csv("metadata.csv", index=False)
gt.to_csv("data/labels/gt_counts.csv", index=False)

# ringkasan
d = meta[meta["status"] == "dipakai"].merge(sp[["filename", "split", "domain"]], on="filename")
print("\nJumlah buah per foto, per split:")
print(d.groupby("split")["jumlah_gt"].describe()[["count", "min", "50%", "max", "mean"]])
print("\nPer domain:")
print(d.groupby("domain")["jumlah_gt"].describe()[["count", "min", "50%", "max", "mean"]])
print("\nTotal box:", int(d["jumlah_gt"].sum()))

print("\nMasalah yang perlu dicek:")
for fn, msg in masalah:
    print(f" - {fn}: {msg}")
if not masalah:
    print(" (tidak ada)")