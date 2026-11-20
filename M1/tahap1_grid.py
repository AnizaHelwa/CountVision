import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

df = pd.read_csv("metadata.csv")
n = len(df)
cols = 6
rows = (n + cols - 1) // cols

fig, axes = plt.subplots(rows, cols, figsize=(18, 3 * rows))
for ax in axes.ravel():
    ax.axis("off")
for ax, fn in zip(axes.ravel(), df["filename"]):
    ax.imshow(Image.open(f"data/raw/{fn}"))
    ax.set_title(fn.replace("grape_", "").replace(".jpg", ""), fontsize=9)
plt.tight_layout()
plt.savefig("grid_semua_foto.png", dpi=100)
plt.show()