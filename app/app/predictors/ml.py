"""Mode 2: Machine Learning klasik (SVM / Random Forest), memakai kode dan model M2 apa adanya.

Alur: kandidat dari pipeline M1 (HSV final + B1) -> M2/inferensi_ml.prediksi_ml (21 fitur HSV, geometri,
LBP; lalu SVM / Random Forest) -> kandidat berlabel 1 = anggur -> jumlah anggur.

Yang dipakai dari folder M2/ (lihat M2/INTEGRASI_WEB.md):
  - M2/inferensi_ml.py : load_model(nama), prediksi_ml(img, mask, boxes, model)
  - M2/results/svm_model.pkl dan rf_model.pkl
Gambar, mask, dan kotak harus pada ukuran yang sama dengan saat training M2: gambar 1024 px (hasil resize
M1), mask hasil morfologi, kotak dalam piksel. Itu yang dikirim oleh pcd.predict().
"""
import sys
from pathlib import Path

from . import pcd
from .common import Hasil, ModelBelumTersedia

M2_DIR = Path(__file__).resolve().parents[2] / "M2"
BERKAS = {"SVM": "svm_model.pkl", "Random Forest": "rf_model.pkl"}      # nama = argumen load_model M2
MODELS = list(BERKAS)
_cache = {}


def tersedia(nama):
    return (M2_DIR / "inferensi_ml.py").exists() and (M2_DIR / "results" / BERKAS[nama]).exists()


def _modul_m2():
    if str(M2_DIR) not in sys.path:
        sys.path.append(str(M2_DIR))        # di belakang M1 supaya nama modul M1 tidak tertimpa
    import inferensi_ml
    return inferensi_ml


def predict(img_bgr, nama="SVM"):
    if not tersedia(nama):
        raise ModelBelumTersedia(f"Mode ML ({nama}) belum tersedia: M2/inferensi_ml.py atau "
                                 f"M2/results/{BERKAS[nama]} tidak ditemukan.")
    try:
        m2 = _modul_m2()
        if nama not in _cache:
            _cache[nama] = m2.load_model(nama)
    except ImportError as e:
        raise ModelBelumTersedia(f"Gagal memuat kode M2 (library kurang?): {e}")

    h = pcd.predict(img_bgr, "HSV (final)", "B1")                 # kandidat + mask dari M1
    lolos = m2.prediksi_ml(h.img, h.mask, h.boxes_px, _cache[nama])
    ht, wd = h.mask.shape
    norm = [(x1 / wd, y1 / ht, x2 / wd, y2 / ht) for x1, y1, x2, y2 in lolos]
    return Hasil(img=h.img, boxes=norm, mask=h.mask, boxes_px=list(lolos),
                 info={"model": nama, "kandidat": len(h.boxes_px), "lolos": len(lolos)})
