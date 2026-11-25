"""CountVision: aplikasi web penghitung anggur. Jalankan dari ROOT repositori:  streamlit run app/app.py"""
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageOps

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "M1"))     # preprocess, counting, grape_pipeline
sys.path.insert(0, str(HERE))                    # paket predictors

from predictors import ml, pcd, yolo                                   # noqa: E402
from predictors.common import ModelBelumTersedia, gambar_hasil         # noqa: E402

st.set_page_config(page_title="CountVision", page_icon="🍇", layout="wide")
st.title("🍇 CountVision: Penghitung Anggur")
st.caption("Unggah foto anggur, pilih mode, lalu sistem mendeteksi dan menghitung jumlah buahnya.")

# ---------------- sidebar ----------------
MODE1, MODE2, MODE3 = "1. Pengolahan citra (PCD)", "2. Machine learning (SVM / Random Forest)", "3. YOLO"
mode = st.sidebar.radio("Mode", [MODE1, MODE2, MODE3])
opsi = {}
if mode == MODE1:
    opsi["metode"] = st.sidebar.selectbox("Metode segmentasi", list(pcd.METODE))
    opsi["penghitungan"] = st.sidebar.radio(
        "Cara menghitung", ["B1", "B0"],
        format_func=lambda x: "B1: watershed (memisah buah menempel)" if x == "B1" else "B0: gumpalan mask")
elif mode == MODE2:
    opsi["nama"] = st.sidebar.radio("Model", ml.MODELS)
    st.sidebar.caption("Kandidat dari pipeline PCD, lalu disaring oleh model M2." +
                       ("" if ml.tersedia(opsi["nama"]) else "  \n**Model belum tersedia.**"))
else:
    opsi["conf"] = st.sidebar.slider("Ambang kepercayaan", 0.05, 0.95, yolo.CONF_DEFAULT, 0.05)
    opsi["imgsz"] = st.sidebar.select_slider("Ukuran input", [320, 480, 640, 800, 1024], yolo.IMGSZ_DEFAULT)
    opsi["varian"] = st.sidebar.selectbox("Preprocessing sebelum YOLO", ["P0", "P1", "P2", "P3", "P4"],
                                          help="Samakan dengan data yang dipakai M3 saat training.")
    if not yolo.tersedia():
        st.sidebar.caption("**Model belum tersedia.**")

# ---------------- unggah gambar ----------------
file = st.file_uploader("Unggah foto anggur", type=["jpg", "jpeg", "png"])
if file is None:
    st.info("Belum ada gambar. Unggah foto anggur untuk memulai.")
    st.stop()

pil = ImageOps.exif_transpose(Image.open(file)).convert("RGB")      # koreksi orientasi foto HP
img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

try:
    t0 = time.perf_counter()
    if mode == MODE1:
        hasil = pcd.predict(img, **opsi)
    elif mode == MODE2:
        hasil = ml.predict(img, **opsi)
    else:
        hasil = yolo.predict(img, **opsi)
    detik = time.perf_counter() - t0
except ModelBelumTersedia as e:
    st.warning(str(e))
    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Gambar yang diunggah")
    st.stop()

# ---------------- tampilkan hasil ----------------
hasil_img = gambar_hasil(hasil.img, hasil.boxes)
kiri, kanan = st.columns(2)
kiri.image(cv2.cvtColor(hasil.img, cv2.COLOR_BGR2RGB), caption="Gambar masukan")
kanan.image(cv2.cvtColor(hasil_img, cv2.COLOR_BGR2RGB), caption="Hasil deteksi")

m1, m2 = st.columns(2)
m1.metric("Jumlah anggur terdeteksi", hasil.count)
m2.metric("Waktu proses", f"{detik:.2f} detik")

if hasil.mask is not None:
    with st.expander("Lihat mask segmentasi"):
        st.image(hasil.mask, caption="Mask (putih = piksel yang dianggap anggur)")
with st.expander("Detail"):
    st.json(hasil.info)

ok, png = cv2.imencode(".png", hasil_img)
if ok:
    st.download_button("Unduh gambar hasil", png.tobytes(), "hasil_countvision.png", "image/png")

st.caption("Catatan: hasil PCD paling lemah pada tandan padat dan latar berwarna ungu (mirip warna anggur).")
