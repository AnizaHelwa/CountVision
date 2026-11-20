import os, glob, random
import cv2

SPLIT = "test"   # ganti: train / valid / test
base = f"data/labels/yolo/{SPLIT}"
os.makedirs("cek_label", exist_ok=True)

imgs = glob.glob(f"{base}/images/*.jpg")
for p in random.sample(imgs, min(8, len(imgs))):
    img = cv2.imread(p)
    h, w = img.shape[:2]
    lbl = f"{base}/labels/" + os.path.splitext(os.path.basename(p))[0] + ".txt"
    n = 0
    if os.path.exists(lbl):
        for line in open(lbl):
            parts = line.split()
            if len(parts) < 5:
                continue
            x, y, bw, bh = map(float, parts[1:5])
            x1, y1 = int((x - bw/2) * w), int((y - bh/2) * h)
            x2, y2 = int((x + bw/2) * w), int((y + bh/2) * h)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), max(2, w // 400))
            n += 1
    cv2.putText(img, f"{n} box", (20, 60), cv2.FONT_HERSHEY_SIMPLEX,
                max(1, w / 1000), (0, 255, 255), 3)
    cv2.imwrite(f"cek_label/{SPLIT}_{os.path.basename(p)}", img)
print("Selesai, lihat folder cek_label/")