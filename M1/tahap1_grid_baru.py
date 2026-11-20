import os, math
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

MULAI = 37          # mulai dari grape_037
PER_SHEET, COLS = 12, 4
os.makedirs("grid_baru", exist_ok=True)

df = pd.read_csv("metadata.csv", sep=None, engine="python")
df["status"] = df["status"].fillna("dipakai")
df = df[df["status"] == "dipakai"].copy()
df["no"] = df["filename"].str.extract(r"(\d+)")[0].astype(int)
df = df[df["no"] >= MULAI].sort_values("no").reset_index(drop=True)

n_sheet = math.ceil(len(df) / PER_SHEET)
for s in range(n_sheet):
    chunk = df.iloc[s * PER_SHEET:(s + 1) * PER_SHEET]
    rows = math.ceil(len(chunk) / COLS)
    fig, axes = plt.subplots(rows, COLS, figsize=(16, 4.3 * rows))
    axes = axes.ravel()
    for ax in axes:
        ax.axis("off")
    for ax, (_, r) in zip(axes, chunk.iterrows()):
        img = Image.open(f"data/raw/{r['filename']}")
        img.draft("RGB", (700, 700))
        img.thumbnail((700, 700))
        ax.imshow(img)
        asli = os.path.splitext(str(r["nama_asli"]))[0]
        ax.set_title(f"{r['no']:03d}  ({asli})", fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = f"grid_baru/grid_{s + 1:02d}.png"
    fig.savefig(out, dpi=70)
    plt.close(fig)
    print(f"{out}: foto {chunk['no'].iloc[0]} sampai {chunk['no'].iloc[-1]}")