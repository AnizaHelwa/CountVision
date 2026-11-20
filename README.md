# CountVision: Deteksi dan Penghitungan Anggur dalam Satu Citra

Proyek mata kuliah **Pengenalan Citra Digital dan Computer Vision**. Sistem ini menghitung jumlah anggur merah-ungu dalam satu foto, dengan variasi posisi, jumlah, latar, dan pencahayaan.

**Metrik utama:** MAE penghitungan, Precision, Recall (ditambah F1 dan MAE relatif).

## Tim dan pembagian kerja

| Folder | Penanggung jawab | Isi |
|---|---|---|
| `M1/` | [nama M1] | Akuisisi dan pelabelan data, EDA, preprocessing, segmentasi, penghitungan klasik (baseline) |
| `M2/` | [nama M2] | Ekstraksi fitur dan ML klasik |
| `M3/` | [nama M3] | Model DL (YOLO) dan prototipe |

## Struktur repositori

```
CountVision/
├── M1/    pengolahan citra dan baseline klasik (dijelaskan di bawah)
├── M2/    [diisi M2]
├── M3/    [diisi M3]
└── README.md
```

Foto dan dataset **tidak ada di repositori** karena ukurannya besar. Dataset dibagikan lewat Google Drive: [tautan Drive].

---

# Bagian M1: Akuisisi Data dan Pengolahan Citra

## Dataset

| Aspek | Isi |
|---|---|
| Foto dipakai | 108 (dua foto grayscale hasil edit dikeluarkan) |
| Scene independen | 51 (foto dari susunan anggur yang sama dihitung satu scene) |
| Domain | 33 foto buah lepas, 75 foto tandan (18 scene) |
| Latar | 11 jenis: putih glossy dan polos, piring ungu, handuk kuning dan peach, talenan kayu, meja kayu, granit gelap, batu abu, kain motif, meja gelap |
| Pencahayaan | merata 69, spotlight 35, redup 4 |
| Label | bounding box, satu kelas `grape`, 1.387 box (Roboflow, ekspor YOLO) |
| Aturan label | buah terlihat ≥ 50% dilabel; tangkai dan bayangan tidak ikut (`M1/labeling_rules.md`) |

**Pembagian data dilakukan per scene**, bukan per foto, supaya foto yang hampir kembar tidak terpecah ke dua split.

| Split | Foto | Scene | Rata-rata buah/foto | Maks |
|---|---|---|---|---|
| Train | 61 | 34 | 11,5 | 28 |
| Val | 23 | 8 | 11,3 | 21 |
| Test | 24 | 9 | 17,9 | 28 |

Daftar split ada di `M1/data/splits/split.csv`, dan atribut tiap foto (latar, pencahayaan, sudut, jarak, scene) di `M1/metadata.csv`.

## Pipeline final

```
foto -> resize (sisi terpanjang 1024 px) -> segmentasi HSV -> morfologi -> penghitungan B1 -> ROI (kotak per buah)
```

| Tahap | Metode | Parameter final |
|---|---|---|
| Resize | Interpolasi `INTER_AREA`, rasio aspek dijaga | sisi terpanjang 1024 px |
| Segmentasi | Threshold HSV, dua rentang hue (merah ada di dua ujung skala hue) | hue 0-12 dan 130-179, S ≥ 120, V ≥ 15 |
| Morfologi | Opening, closing, penutupan lubang kecil | opening 9 px, closing 7 px, lubang < 400 px² |
| **B0** | Connected components (satu gumpalan mask = satu buah) | `min_area` 1000 px² |
| **B1** | Distance transform, puncak lokal, watershed (memisahkan buah yang menempel) | jarak minimum 45 px, puncak minimum 10 px, `min_area` 1000 px² |

Konfigurasi final adalah **P0 + B1**.

## Protokol evaluasi

- Kotak prediksi dicocokkan satu-lawan-satu dengan label (IoU ≥ 0,5 dan ≥ 0,3).
- Parameter hanya di-tuning di **train**; val untuk membandingkan konfigurasi; **test dijalankan sekali** di akhir untuk konfigurasi final.
- Kriteria seleksi: F1@0,3. Seleksi dengan MAE saja ternyata menyesatkan (pecahan mask kecil terhitung sebagai buah dan menutupi kekurangan hitung di tandan padat), jadi F1 selalu dilaporkan bersama MAE.

## Hasil

**Baseline (B1 mengungguli B0)**

| | MAE val | F1@0,3 val | MAE test | MAE relatif test | F1@0,3 test |
|---|---|---|---|---|---|
| B0 | 5,44 | 0,51 | 10,46 | 0,585 | 0,324 |
| **B1** | **3,48** | **0,66** | **6,50** | **0,364** | **0,541** |

**Perbandingan preprocessing dan metode segmentasi (B1, data val; eksperimen eksploratif)**

| | Isi | MAE | F1@0,3 |
|---|---|---|---|
| **P0** | resize + HSV (final) | **3,48** | **0,662** |
| P1 | + Gaussian 5×5 | 3,65 | 0,621 |
| P2 | + CLAHE | 4,96 | 0,528 |
| P3 | + gamma 1,4, CLAHE, Gaussian | 3,96 | 0,569 |
| P4 | + gamma, CLAHE, bilateral, sharpening | 4,52 | 0,564 |
| Otsu | ambang otomatis pada kanal Saturation | 5,70 | 0,602 |
| K-means | klaster warna di ruang LAB (k=4) | 4,74 | 0,525 |

**Temuan utama**

1. Distance transform dan watershed memangkas MAE hampir separuh dibanding connected components saja.
2. Preprocessing klasik (CLAHE, gamma, filter) tidak memperbaiki penghitungan; CLAHE bahkan memperburuk foto spotlight. Gamma dan parameter CLAHE dipatok, tidak di-tuning.
3. HSV paling stabil antara train dan val; Otsu dan K-means membaik di train tetapi turun di val.
4. Kesalahan di test terkonsentrasi di tiga scene (sekitar 97% total error): **kurang hitung** pada tandan padat (kulit pucat dan buah gelap hilang dari mask) dan **kelebihan hitung** di piring ungu (dinding piring punya hue dan saturation mirip anggur).

**Keterbatasan:** val dan test kecil (8 dan 9 scene), beberapa kategori hanya satu scene, val dipakai berulang untuk keputusan desain sehingga angkanya sedikit optimis, dan test hanya dijalankan untuk konfigurasi final.

## Isi folder `M1/`

| File atau folder | Fungsi |
|---|---|
| `preprocess.py` | Resize, gamma, CLAHE, filter, segmentasi HSV (dan LAB), morfologi; `FINAL_CONFIG` |
| `counting.py` | Penghitungan B0 dan B1 |
| `evaluate.py` | MAE, Precision, Recall, F1, pencocokan IoU |
| `grape_pipeline.py` | Pipeline final `detect_grapes()` dan ekspor varian dataset untuk M3 |
| `tahap4_baseline.py` | Tuning di train, evaluasi val, mode `--test` |
| `tahap5_pipeline.py` | Perbandingan preprocessing P0-P4 |
| `tahap6_segmentasi.py` | Perbandingan Otsu dan K-means |
| `tahap1_*.py`, `tahap2_*.py`, `tahap3_*.py` | Penataan foto dan metadata, split, hitung jumlah GT, EDA |
| `metadata.csv`, `data/splits/split.csv`, `labeling_rules.md` | Metadata, pembagian data, aturan pelabelan |
| `data/labels/` | Label YOLO (file `.txt`) |
| `eda_output/`, `hasil_*/` | Gambar EDA dan hasil percobaan (CSV dan JSON) |

## Cara menjalankan

Semua perintah dijalankan **dari dalam folder `M1/`**, karena path bersifat relatif.

```powershell
cd M1
python -m venv venv
venv\Scripts\activate
pip install opencv-python numpy pandas matplotlib pillow scikit-image
```

Taruh data dari Google Drive di `M1/`: foto di `data/raw/` dan ekspor label YOLO (Roboflow) di `data/labels/yolo/`.

| Tujuan | Perintah |
|---|---|
| Tuning di train, evaluasi di train dan val | `python tahap4_baseline.py --space hsv` |
| Perbandingan preprocessing P0-P4 | `python tahap5_pipeline.py` |
| Perbandingan Otsu dan K-means | `python tahap6_segmentasi.py` |
| Ekspor varian dataset YOLO (untuk M3) | `python grape_pipeline.py --ekspor P0 P2 P3` |
| Evaluasi di test (**sekali saja**) | `python tahap4_baseline.py --space hsv --test` |

Mode `--test` menulis file kunci, sehingga tidak bisa dijalankan ulang tanpa sengaja. Jangan memakai test untuk memilih parameter atau model.

## Memakai pipeline dari kode lain

```python
import cv2
from grape_pipeline import detect_grapes

img = cv2.imread("foto.jpg")
boxes, mask = detect_grapes(img)      # boxes: (x1, y1, x2, y2) pada gambar yang sudah di-resize
print("Jumlah anggur:", len(boxes))

boxes_norm, _ = detect_grapes(img, normalized=True)   # kotak berskala 0-1
```

---

# Bagian M2: Ekstraksi Fitur dan ML Klasik

[Diisi oleh M2: fitur yang dipakai, metode ML, hasil, confusion matrix, cara menjalankan.]

# Bagian M3: Model DL dan Integrasi

[Diisi oleh M3: model, perbandingan raw vs preprocessed, hasil, cara menjalankan prototipe.]

---

## Catatan untuk seluruh tim

- Pembagian data di `M1/data/splits/split.csv` bersifat tetap (per scene). Jangan diacak ulang.
- Test dipakai hanya untuk evaluasi akhir.
- Foto, dataset, dan model berukuran besar tidak diunggah ke repositori; bagikan lewat Google Drive.
