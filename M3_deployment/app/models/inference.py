import importlib.util
from pathlib import Path

import cv2
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("countvision_preprocess", ROOT / "preprocess_m1.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def load_model():
    return YOLO(str(ROOT / "best.pt"))


def predict_bgr(model, image, already_p4=False):
    if image is None:
        raise ValueError("Gambar tidak dapat dibaca.")
    if not already_p4:
        config = module.get_config(
            gamma=1.4,
            clahe={"clip": 2.0, "tile": 8},
            denoise={"method": "bilateral", "k": 9},
            sharpen={"sigma": 1.5, "amount": 0.7},
        )
        image = module.preprocess(image, config)
        success, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if not success:
            raise RuntimeError("Gagal menyiapkan gambar P4.")
        image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    return model.predict(image, imgsz=640, conf=0.50, iou=0.70,
                         max_det=300, device="cpu", verbose=False)[0]
