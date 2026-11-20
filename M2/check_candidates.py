from pathlib import Path

import cv2
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

root = Path(__file__).resolve().parent.parent
result_dir = root / "M2" / "results"
data_dir = root / "yolo_variants" / "P0"

df = pd.read_csv(result_dir / "features.csv")

print("Jumlah kandidat berdasarkan label:")
print(df["label"].value_counts().sort_index())

labels = [
    (1, "Cocok dengan anggur"),
    (0, "Kecocokan rendah"),
    (-1, "Ambigu")
]

fig, axes = plt.subplots(3, 5, figsize=(16, 10))

for row, (label, title) in enumerate(labels):
    samples = df[
        (df["split"] == "train") &
        (df["label"] == label)
    ].sample(n=5, random_state=42)

    for col, (_, item) in enumerate(samples.iterrows()):
        path = data_dir / item["split"] / "images" / item["filename"]
        img = cv2.imread(str(path))

        ax = axes[row, col]

        if img is None:
            ax.set_title("Gambar tidak ditemukan")
            ax.axis("off")
            continue

        h, w = img.shape[:2]
        x1 = max(0, int(item["x1"]) - 35)
        y1 = max(0, int(item["y1"]) - 35)
        x2 = min(w, int(item["x2"]) + 35)
        y2 = min(h, int(item["y2"]) + 35)

        crop = cv2.cvtColor(
            img[y1:y2, x1:x2],
            cv2.COLOR_BGR2RGB
        )

        ax.imshow(crop)

        rect = Rectangle(
            (item["x1"] - x1, item["y1"] - y1),
            item["x2"] - item["x1"],
            item["y2"] - item["y1"],
            fill=False,
            edgecolor="red",
            linewidth=2
        )

        ax.add_patch(rect)
        ax.set_title(f"IoU: {item['iou']:.2f}", fontsize=10)
        ax.axis("off")

    axes[row, 0].set_ylabel(
        title, fontsize=11, rotation=90, labelpad=12
    )

plt.suptitle("Pemeriksaan Kandidat Anggur - Data Training", fontsize=15)
plt.tight_layout()
plt.savefig(
    result_dir / "candidate_check.png",
    dpi=150,
    bbox_inches="tight"
)
plt.show()

print("\nVisualisasi tersimpan di M2/results/candidate_check.png")
