"""Struktur hasil dan fungsi gambar yang dipakai ketiga mode."""
from dataclasses import dataclass, field

import cv2
import numpy as np


class ModelBelumTersedia(Exception):
    """Dilempar saat model sebuah mode belum diserahkan (M2/M3 masih dalam proses)."""


@dataclass
class Hasil:
    img: np.ndarray                      # gambar (BGR) yang dipakai untuk menggambar kotak
    boxes: list                          # (x1, y1, x2, y2) skala 0-1 terhadap img
    mask: np.ndarray = None              # opsional: mask segmentasi (mode PCD)
    boxes_px: list = None                # opsional: kotak dalam piksel pada img (dipakai mode ML)
    info: dict = field(default_factory=dict)

    @property
    def count(self):
        return len(self.boxes)


def gambar_hasil(img_bgr, boxes, warna=(0, 0, 255)):
    out = img_bgr.copy()
    h, w = out.shape[:2]
    tebal = max(2, w // 300)
    for x1, y1, x2, y2 in boxes:
        cv2.rectangle(out, (int(x1 * w), int(y1 * h)), (int(x2 * w), int(y2 * h)), warna, tebal)
    return out
