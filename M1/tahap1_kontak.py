import os, math
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

PER_SHEET, COLS = 20, 5
os.makedirs("contact_sheets", exist_ok=True)

df = pd.read_csv("metadata.csv", sep=None, engine="python")
df["status"] = df["status"].fillna("dipakai")
df = df[df["status"] == "dipakai"].reset_index(drop=True)

n_sheet = math.ceil(len(df) / PER_SHEET)
for s in range(n_sheet):
    chunk = df.iloc[s * PER_SHEET:(s + 1) * PER_SHEET]
    rows = math.ceil(len(chunk) / COLS)
    fig, axes = plt.subplots(rows, COLS, figsize=(20, 4.2 * rows))
    axes = axes.ravel()
    for ax in axes:
        ax.axis("off")
    nomor = []
    for ax, fn in zip(axes, chunk["filename"]):
        img = Image.open(f"data/raw/{fn}")
        img.draft("RGB", (500, 500))
        img.thumbnail((500, 500))
        ax.imshow(img)
        n = int(fn[6:9])
        nomor.append(n)
        ax.set_title(str(n), fontsize=16, fontweight="bold")
    plt.tight_layout()
    out = f"contact_sheets/sheet_{s + 1:02d}.png"
    fig.savefig(out, dpi=80)
    plt.close(fig)
    print(f"{out}: foto {nomor[0]} sampai {nomor[-1]}")

# template tabel scene (tidak menimpa kalau sudah ada)
if not os.path.exists("scene_table.csv"):
    pd.DataFrame(columns=["scene_id", "foto", "background", "pencahayaan",
                          "posisi", "sudut", "jarak", "sumber", "catatan"]
                 ).to_csv("scene_table.csv", index=False)
    print("scene_table.csv dibuat")