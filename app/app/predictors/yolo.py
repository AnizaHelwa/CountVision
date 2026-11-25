"""Mode 3: YOLO -- DIISI OLEH M3.

Yang perlu dilakukan M3:
  1. Taruh bobot terbaik di app/models/best.pt
  2. Kalau modelnya dilatih pada gambar yang sudah diproses (misalnya P2 atau P3), pilih varian
     yang sama di sidebar aplikasi, karena preprocessing yang sama harus dijalankan saat inferensi.
  3. Atur CONF_DEFAULT dan IMGSZ_DEFAULT sesuai hasil evaluasi.
Belum diuji dengan model sungguhan (ultralytics belum terpasang saat kerangka ini dibuat).
"""
from functools import lru_cache
from pathlib import Path

import preprocess as pp                    # dari folder M1/
from grape_pipeline import VARIANTS         # varian preprocessing P0-P4

from .common import Hasil, ModelBelumTersedia

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "best.pt"
CONF_DEFAULT, IMGSZ_DEFAULT = 0.25, 640


def tersedia():
    return MODEL_PATH.exists()


@lru_cache(maxsize=1)
def _muat():
    from ultralytics import YOLO
    return YOLO(str(MODEL_PATH))


def predict(img_bgr, conf=CONF_DEFAULT, imgsz=IMGSZ_DEFAULT, varian="P0"):
    if not MODEL_PATH.exists():
        raise ModelBelumTersedia("Mode YOLO belum tersedia: menunggu file bobot dari M3 "
                                 "(taruh di app/models/best.pt).")
    try:
        model = _muat()
    except ImportError:
        raise ModelBelumTersedia("Paket 'ultralytics' belum terpasang (pip install ultralytics).")
    img = pp.preprocess(img_bgr, pp.get_config(**VARIANTS[varian]))
    res = model.predict(img, conf=conf, imgsz=imgsz, verbose=False)[0]
    boxes = [tuple(float(v) for v in b) for b in res.boxes.xyxyn.cpu().numpy()]
    return Hasil(img=img, boxes=boxes, info={"conf": conf, "imgsz": imgsz, "varian": varian})
