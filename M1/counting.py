"""counting.py - mengubah mask biner menjadi daftar kotak prediksi.

B0: connected components (contour/CCL). Buah yang menempel menjadi satu blob.
B1: distance transform + peak + watershed, untuk memisahkan buah yang menempel.

Semua kotak dikembalikan dalam piksel, format (x1, y1, x2, y2).
"""
import cv2
import numpy as np
from skimage.measure import regionprops
from skimage.segmentation import watershed


def count_b0(mask, min_area=300):
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    boxes = []
    for i in range(1, n):
        x, y, w, h, a = stats[i]
        if a >= min_area:
            boxes.append((int(x), int(y), int(x + w), int(y + h)))
    return boxes


def count_b1(mask, min_dist=25, min_peak=8, min_area=300, smooth=5):
    """min_dist : jarak minimum antar puncak (px), kira-kira radius buah terkecil.
    min_peak : nilai distance transform minimum agar sebuah puncak dianggap buah (px).
    smooth   : lebar Gaussian untuk menghaluskan peta jarak (0 = tanpa)."""
    binary = mask > 0
    if not binary.any():
        return []

    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5).astype(np.float32)
    if smooth and smooth > 0:
        dist = cv2.GaussianBlur(dist, (0, 0), smooth / 2.0)

    k = 2 * int(min_dist) + 1
    local_max = dist == cv2.dilate(dist, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    peaks = (local_max & (dist >= min_peak) & binary).astype(np.uint8)
    n_peak, markers = cv2.connectedComponents(peaks, connectivity=8)   # puncak = 1..n_peak-1

    boxes = []
    ws = np.zeros(mask.shape, np.int32)
    if n_peak > 1:
        ws = watershed(-dist, markers, mask=binary).astype(np.int32)
        for rp in regionprops(ws):
            if rp.area >= min_area:
                r0, c0, r1, c1 = rp.bbox
                boxes.append((int(c0), int(r0), int(c1), int(r1)))

    # blob yang tidak punya puncak (terlalu kecil/tipis) tetap dihitung sebagai satu objek
    leftover = (binary & (ws == 0)).astype(np.uint8) * 255
    boxes += count_b0(leftover, min_area=min_area)
    return boxes


METHODS = {"B0": count_b0, "B1": count_b1}
