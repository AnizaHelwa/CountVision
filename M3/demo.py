import argparse
import sys
from pathlib import Path

import cv2
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))

from preprocess import get_config, preprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("foto", help="Lokasi foto JPG atau PNG")
    parser.add_argument(
        "--sudah-p4",
        action="store_true",
        help="Gunakan untuk foto dari folder yolo_variants/P4",
    )
    args = parser.parse_args()

    source = Path(args.foto)
    image = cv2.imread(str(source))
    if image is None:
        raise ValueError(f"Foto gagal dibaca: {source}")

    if not args.sudah_p4:
        config = get_config(
            gamma=1.4,
            clahe={"clip": 2.0, "tile": 8},
            denoise={"method": "bilateral", "k": 9},
            sharpen={"sigma": 1.5, "amount": 0.7},
        )
        image = preprocess(image, config)

        # Samakan kompresi dengan gambar ekspor P4 dari M1.
        success, encoded = cv2.imencode(
            ".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 95]
        )
        if not success:
            raise RuntimeError("Gagal menyiapkan gambar P4.")
        image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)

    model = YOLO(str(ROOT / "models" / "P4_best.pt"))
    result = model.predict(
        source=image,
        conf=0.50,
        iou=0.70,
        imgsz=640,
        max_det=300,
        device="cpu",
        verbose=False,
    )[0]

    count = len(result.boxes)
    annotated = result.plot()

    # Tambahkan ruang untuk judul agar kotak deteksi tetap terlihat.
    annotated = cv2.copyMakeBorder(
        annotated, 55, 0, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0)
    )
    cv2.putText(
        annotated,
        f"Total anggur: {count} butir",
        (10, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
    )

    output = ROOT / "demo_output"
    output.mkdir(exist_ok=True)
    destination = output / f"{source.stem}_P4_demo.jpg"

    if not cv2.imwrite(str(destination), annotated):
        raise RuntimeError(f"Gagal menyimpan gambar: {destination}")

    print(f"Total anggur: {count} butir")
    print(f"Gambar tersimpan di: {destination}")


if __name__ == "__main__":
    main()