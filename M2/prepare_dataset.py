from pathlib import Path
import cv2
import pandas as pd
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parent.parent
result_dir = root / "M2" / "results"
data_dir = root / "yolo_variants" / "P0"

review_dir = result_dir / "review"
review_dir.mkdir(exist_ok=True)

df = pd.read_csv(result_dir / "features.csv")

review = df[
    (df["split"] == "train") &
    (df["label"].isin([0, -1]))
].copy().reset_index(drop=True)

review["review_id"] = range(1, len(review) + 1)
review["manual_label"] = ""

def read_labels(path, width, height):
    boxes = []

    if not path.exists():
        raise FileNotFoundError(path)

    for line in path.read_text().splitlines():
        if not line.strip():
            continue

        cls, xc, yc, bw, bh = map(float, line.split())

        if int(cls) != 0:
            continue

        boxes.append((
            int((xc - bw / 2) * width),
            int((yc - bh / 2) * height),
            int((xc + bw / 2) * width),
            int((yc + bh / 2) * height)
        ))

    return boxes


for start in range(0, len(review), 20):
    batch = review.iloc[start:start + 20]
    fig, axes = plt.subplots(4, 5, figsize=(16, 13))
    axes = axes.flatten()

    for ax in axes:
        ax.axis("off")

    for ax, (_, item) in zip(axes, batch.iterrows()):
        path = data_dir / "train" / "images" / item["filename"]
        label_path = (
            data_dir / "train" / "labels" /
            f"{Path(item['filename']).stem}.txt"
        )

        img = cv2.imread(str(path))

        if img is None:
            ax.set_title("Gambar tidak ditemukan")
            continue

        h, w = img.shape[:2]
        gt_boxes = read_labels(label_path, w, h)

        x1, y1, x2, y2 = [
            int(item[k]) for k in ["x1", "y1", "x2", "y2"]
        ]

        # Kandidat segmentasi
        cv2.rectangle(
            img, (x1, y1), (x2, y2),
            (0, 0, 255), 3
        )

        # Ground truth
        for gx1, gy1, gx2, gy2 in gt_boxes:
            cv2.rectangle(
                img, (gx1, gy1), (gx2, gy2),
                (0, 255, 0), 2
            )

        pad = 90
        cx1 = max(0, x1 - pad)
        cy1 = max(0, y1 - pad)
        cx2 = min(w, x2 + pad)
        cy2 = min(h, y2 + pad)

        crop = img[cy1:cy2, cx1:cx2]

        if crop.size == 0:
            continue

        ax.imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        ax.set_title(
            f"ID {item['review_id']} | "
            f"Label {item['label']} | "
            f"IoU {item['iou']:.2f}",
            fontsize=9
        )
        ax.axis("off")

    page = start // 20 + 1
    plt.tight_layout()

    plt.savefig(
        review_dir / f"review_{page:02d}.png",
        dpi=120
    )
    plt.close()

review[[
    "review_id", "filename", "candidate_id",
    "label", "iou", "manual_label"
]].to_csv(
    review_dir / "review_labels.csv",
    index=False
)

print("Jumlah kandidat untuk diperiksa:", len(review))
print("Jumlah halaman:", (len(review) + 19) // 20)
print("Hasil tersimpan di:", review_dir)
