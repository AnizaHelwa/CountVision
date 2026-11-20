import os, re, glob
import numpy as np
import pandas as pd
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LABEL_DIR = "data/labels/yolo"
RF2SPLIT = {"train": "train", "valid": "val", "test": "test"}
MAX_SIDE = 1024        # ukuran kerja, sama dengan rencana resize
N_SAMPLE = 2000        # piksel per kelas per foto
OUT = "eda_output"
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(0)

# ---------- data ----------
meta = pd.read_csv("metadata.csv")
meta = meta[meta["status"] == "dipakai"].drop(columns=["split", "domain"], errors="ignore")
sp = pd.read_csv("data/splits/split.csv")
df = meta.merge(sp[["filename", "split", "domain"]], on="filename")

def load_boxes():
    d = {}
    for rf in RF2SPLIT:
        for p in glob.glob(f"{LABEL_DIR}/{rf}/labels/*.txt"):
            m = re.search(r"(grape_\d{3})", os.path.basename(p))
            if not m:
                continue
            b = []
            for line in open(p):
                s = line.split()
                if len(s) >= 5:
                    b.append(tuple(map(float, s[1:5])))
            d[m.group(1) + ".jpg"] = b
    return d

boxes_all = load_boxes()

# ---------- loop per foto ----------
img_rows, box_rows, samples = [], [], []
for _, r in df.iterrows():
    fn = r["filename"]
    img = cv2.imread(f"data/raw/{fn}")
    if img is None:
        print("GAGAL baca:", fn)
        continue
    h0, w0 = img.shape[:2]
    s = min(1.0, MAX_SIDE / max(h0, w0))
    img = cv2.resize(img, (int(w0 * s), int(h0 * s)), interpolation=cv2.INTER_AREA)
    H, W = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    boxes = boxes_all.get(fn, [])

    img_rows.append(dict(
        filename=fn, split=r["split"], domain=r["domain"],
        pencahayaan=r["pencahayaan"], background=r["background"],
        lebar=w0, tinggi=h0, aspek=round(w0 / h0, 3),
        jumlah_gt=len(boxes),
        mean_V=hsv[..., 2].mean(), mean_S=hsv[..., 1].mean(),
        std_gray=gray.std(), persen_gelap=(gray < 50).mean() * 100))

    gm = np.zeros((H, W), np.uint8)       # mask dalam anggur (elips menyusut 80%)
    ex = np.zeros((H, W), np.uint8)       # area box (untuk dikeluarkan dari latar)
    for x, y, bw, bh in boxes:
        box_rows.append(dict(filename=fn, split=r["split"], domain=r["domain"],
                             diameter_px=np.sqrt(bw * W * bh * H)))
        cx, cy = int(x * W), int(y * H)
        ax, ay = max(int(bw * W / 2 * 0.8), 1), max(int(bh * H / 2 * 0.8), 1)
        cv2.ellipse(gm, (cx, cy), (ax, ay), 0, 0, 360, 255, -1)
        cv2.rectangle(ex, (int((x - bw / 2) * W), int((y - bh / 2) * H)),
                          (int((x + bw / 2) * W), int((y + bh / 2) * H)), 255, -1)
    ex = cv2.dilate(ex, np.ones((15, 15), np.uint8))

    for kelas, pix in (("anggur", hsv[gm > 0]), ("latar", hsv[ex == 0])):
        if len(pix) == 0:
            continue
        pix = pix[rng.choice(len(pix), min(N_SAMPLE, len(pix)), replace=False)]
        samples.append(pd.DataFrame(pix, columns=["H", "S", "V"]).assign(
            kelas=kelas, filename=fn, split=r["split"],
            background=r["background"], pencahayaan=r["pencahayaan"]))

I = pd.DataFrame(img_rows)
B = pd.DataFrame(box_rows)
S = pd.concat(samples, ignore_index=True)
I.to_csv(f"{OUT}/eda_per_foto.csv", index=False)

# ---------- 1. resolusi ----------
print("=== RESOLUSI ===")
print(I.groupby(["lebar", "tinggi"]).size().sort_values(ascending=False).head(8))
print("Aspek rasio min/maks:", I["aspek"].min(), I["aspek"].max())
print("Potret:", (I["tinggi"] > I["lebar"]).sum(), " Lanskap:", (I["lebar"] >= I["tinggi"]).sum())

# ---------- 2. jumlah buah ----------
print("\n=== JUMLAH BUAH PER FOTO ===")
print(I.groupby("domain")["jumlah_gt"].describe()[["count", "min", "50%", "max", "mean"]])

# ---------- 3. kecerahan ----------
print("\n=== KECERAHAN PER PENCAHAYAAN ===")
print(I.groupby("pencahayaan")[["mean_V", "std_gray", "persen_gelap"]].agg(["mean", "std"]).round(1))

# ---------- 4. ukuran buah ----------
print("\n=== DIAMETER BUAH (px, gambar 1024) ===")
print(B.groupby("domain")["diameter_px"].describe()[["count", "min", "25%", "50%", "75%", "max"]].round(1))
med_d = B["diameter_px"].median()
print(f"Median diameter {med_d:.0f}px -> kernel morfologi awal sekitar {max(3, int(med_d * 0.15)) | 1}px, "
      f"min_distance watershed sekitar {int(med_d * 0.5)}px")

# ---------- 5. HSV (rentang HANYA dari train) ----------
tr = S[(S["kelas"] == "anggur") & (S["split"] == "train")]
hs = (tr["H"].astype(int) + 90) % 180          # geser hue supaya merah tidak terpotong
lo_s, hi_s = np.percentile(hs, [2, 98])
s_lo, v_lo = np.percentile(tr["S"], 2), np.percentile(tr["V"], 2)
lo_h, hi_h = int((lo_s - 90) % 180), int((hi_s - 90) % 180)

print("\n=== HSV PIKSEL ANGGUR (train) ===")
print(tr[["H", "S", "V"]].describe(percentiles=[.02, .25, .5, .75, .98]).round(1))
if lo_h <= hi_h:
    print(f"Usulan Hue (p2-p98): {lo_h}..{hi_h}")
else:
    print(f"Usulan Hue melingkar: 0..{hi_h} DAN {lo_h}..179 (dua rentang inRange digabung)")
print(f"S >= {s_lo:.0f}, V >= {v_lo:.0f}  (ini titik awal, bukan nilai final)")

def in_range(d):
    hh = (d["H"].astype(int) + 90) % 180
    return (hh >= lo_s) & (hh <= hi_s) & (d["S"] >= s_lo) & (d["V"] >= v_lo)

S["lolos"] = in_range(S)
print("\n% piksel LATAR yang lolos rentang anggur (makin kecil makin baik), per background:")
print((S[S["kelas"] == "latar"].groupby("background")["lolos"].mean() * 100).round(1).sort_values(ascending=False))
print("\n% piksel ANGGUR yang lolos rentang, per pencahayaan (semua split):")
print((S[S["kelas"] == "anggur"].groupby("pencahayaan")["lolos"].mean() * 100).round(1))

# ---------- gambar ----------
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].scatter(I["lebar"], I["tinggi"], alpha=.6); ax[0].set_title("Resolusi (lebar vs tinggi)")
ax[1].hist(I["aspek"], bins=15); ax[1].set_title("Aspek rasio")
fig.tight_layout(); fig.savefig(f"{OUT}/1_resolusi.png", dpi=110); plt.close(fig)

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for dom in I["domain"].unique():
    ax[0].hist(I[I["domain"] == dom]["jumlah_gt"], bins=range(0, 32, 2), alpha=.6, label=dom)
ax[0].legend(); ax[0].set_title("Jumlah buah per foto")
I.boxplot(column="jumlah_gt", by="split", ax=ax[1]); ax[1].set_title("Per split"); fig.suptitle("")
fig.tight_layout(); fig.savefig(f"{OUT}/2_jumlah_buah.png", dpi=110); plt.close(fig)

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
I.boxplot(column="mean_V", by="pencahayaan", ax=ax[0]); ax[0].set_title("Rata-rata V")
I.boxplot(column="std_gray", by="pencahayaan", ax=ax[1]); ax[1].set_title("Kontras (std gray)"); fig.suptitle("")
fig.tight_layout(); fig.savefig(f"{OUT}/3_kecerahan.png", dpi=110); plt.close(fig)

fig, ax = plt.subplots(1, 3, figsize=(15, 4))
for k, c in (("anggur", "purple"), ("latar", "gray")):
    d = S[(S["kelas"] == k) & ((S["split"] == "train") if k == "anggur" else True)]
    ax[0].hist(d["H"], bins=180, range=(0, 180), density=True, alpha=.55, color=c, label=k)
    ax[1].hist(d["S"], bins=64, range=(0, 255), density=True, alpha=.55, color=c, label=k)
    ax[2].hist(d["V"], bins=64, range=(0, 255), density=True, alpha=.55, color=c, label=k)
for a, t in zip(ax, ["Hue (0-179)", "Saturation", "Value"]):
    a.set_title(t); a.legend()
fig.tight_layout(); fig.savefig(f"{OUT}/4_hsv_anggur_vs_latar.png", dpi=110); plt.close(fig)

fig, ax = plt.subplots(figsize=(6, 5))
ax.hist2d(tr["H"], tr["S"], bins=[90, 64], range=[[0, 180], [0, 256]], cmap="magma", norm=matplotlib.colors.LogNorm())
ax.set_xlabel("Hue"); ax.set_ylabel("Saturation"); ax.set_title("Hue-Saturation piksel anggur (train)")
fig.tight_layout(); fig.savefig(f"{OUT}/5_hue_sat_anggur.png", dpi=110); plt.close(fig)

fig, ax = plt.subplots(figsize=(6, 4))
ax.hist(B["diameter_px"], bins=30); ax.set_title("Diameter buah (px, sisi terpanjang 1024)")
fig.tight_layout(); fig.savefig(f"{OUT}/6_diameter.png", dpi=110); plt.close(fig)

print(f"\nSelesai. Lihat folder {OUT}/")