# CountVision: Deteksi dan Penghitungan Anggur dalam Satu Citra

Proyek mata kuliah **Pengenalan Citra Digital dan Computer Vision**. Sistem ini menghitung jumlah anggur merah-ungu dalam satu foto, dengan variasi posisi, jumlah, latar, dan pencahayaan.

**Metrik utama:** MAE penghitungan, Precision, Recall (ditambah F1 dan MAE relatif).

## Tim dan pembagian kerja

| Folder | Penanggung jawab | Isi |
|---|---|---|
| `M1/` | Aniza Helwa Mahanani | Akuisisi dan pelabelan data, EDA, preprocessing, segmentasi, penghitungan klasik (baseline) |
| `M2/` | Alyaa Rahma Nabilah | Ekstraksi fitur dan ML klasik |
| `M3/` | Ifka Rosmalita | Model DL (YOLO) dan prototipe |

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

# Bagian M2: Ekstraksi Fitur dan Machine Learning Klasik

## Gambaran Umum

Bagian M2 melanjutkan hasil preprocessing dan segmentasi citra dari M1. Kandidat buah anggur yang diperoleh melalui pipeline segmentasi digunakan sebagai input untuk ekstraksi fitur dan klasifikasi menggunakan metode Machine Learning klasik.

Dua model yang diimplementasikan adalah **Support Vector Machine (SVM)** dan **Random Forest**. Tujuan eksperimen ini adalah membandingkan kemampuan kedua model dalam menyaring kandidat hasil segmentasi serta mengevaluasi pengaruhnya terhadap deteksi dan penghitungan buah anggur.

## Ekstraksi Fitur

Ekstraksi fitur dilakukan pada kandidat hasil segmentasi menggunakan tiga kelompok fitur:

| Jenis Fitur | Fitur yang Digunakan | Fungsi |
|---|---|---|
| Warna | Mean dan standar deviasi HSV | Menggambarkan karakteristik warna kandidat |
| Bentuk | Area, perimeter, circularity, aspect ratio, fill ratio | Menggambarkan karakteristik geometris kandidat |
| Tekstur | Histogram Local Binary Pattern (LBP) | Menggambarkan pola tekstur kandidat |

Setiap kandidat direpresentasikan menggunakan **21 fitur numerik**, terdiri dari 6 fitur warna, 5 fitur bentuk, dan 10 fitur tekstur LBP.

Hasil ekstraksi disimpan dalam `M2/results/features.csv`.

## Persiapan Dataset dan Pelabelan

Dataset menggunakan 108 gambar anggur dengan pembagian train, validation, dan test yang mengikuti dataset M1.

| Split | Jumlah Gambar | Anotasi Anggur | Kandidat Segmentasi |
|---|---:|---:|---:|
| Train | 61 | 699 | 594 |
| Validation | 23 | 259 | 213 |
| Test | 24 | 429 | 383 |
| **Total** | **108** | **1.387** | **1.190** |

Pelabelan awal kandidat dilakukan berdasarkan nilai Intersection over Union (IoU) antara bounding box hasil segmentasi dan anotasi ground truth.

| Label | Kriteria | Keterangan |
|---|---|---|
| 1 | IoU ≥ 0,5 | Kecocokan tinggi |
| 0 | IoU ≤ 0,1 | Kecocokan rendah |
| -1 | 0,1 < IoU < 0,5 | Ambigu |

Kandidat dengan label -1 tidak digunakan pada training awal.

Perlu diperhatikan bahwa label tersebut merupakan kategori kecocokan bounding box, bukan label anggur dan non-anggur yang telah diverifikasi secara manual. Kandidat dengan IoU rendah masih mungkin mencakup sebagian objek anggur.

Pemeriksaan visual awal dilakukan melalui `check_candidates.py` dan `prepare_dataset.py`, dengan hasil gambar pemeriksaan pada folder `M2/results/review/`.

## Implementasi Machine Learning

Dua metode Machine Learning digunakan untuk membandingkan hasil klasifikasi kandidat berdasarkan fitur yang telah diekstraksi.

| Model | Konfigurasi |
|---|---|
| Support Vector Machine (SVM) | StandardScaler, kernel RBF, C=1.0, class_weight=balanced |
| Random Forest | 100 decision tree, random_state=42, class_weight=balanced |

Training menggunakan **365 kandidat** berlabel 0 atau 1, sedangkan validation menggunakan **124 kandidat**.

Model dilatih menggunakan fitur warna, bentuk, dan tekstur tanpa memasukkan nilai IoU sebagai fitur input.

## Hasil Evaluasi Validation

Evaluasi validation dilakukan untuk mengukur kemampuan model mengklasifikasikan kategori kecocokan kandidat berdasarkan IoU.

| Metrik | SVM | Random Forest |
|---|---:|---:|
| Accuracy | 95,2% | **96,8%** |
| Precision | 96,6% | **98,2%** |
| Recall | 98,2% | 98,2% |
| F1-score | 97,4% | **98,2%** |

Random Forest memperoleh hasil validation sedikit lebih tinggi dibandingkan SVM.

Berdasarkan confusion matrix, SVM menghasilkan 6 kesalahan klasifikasi dari 124 kandidat, sedangkan Random Forest menghasilkan 4 kesalahan.

Visualisasi confusion matrix tersimpan pada:
- `M2/results/cm_svm.png`
- `M2/results/cm_rf.png`

Hasil ini merupakan evaluasi klasifikasi kategori IoU, bukan akurasi deteksi atau penghitungan buah anggur secara langsung.

## Evaluasi Deteksi dan Counting pada Data Test

Evaluasi dilakukan menggunakan **24 gambar test dengan total 429 anotasi buah anggur**.

Kedua model menerima kandidat dari pipeline segmentasi yang sama. Kandidat yang diprediksi sebagai kelas 1 oleh masing-masing model digunakan sebagai hasil deteksi.

Evaluasi dilakukan menggunakan:
- Precision, Recall, dan F1-score deteksi berdasarkan pencocokan bounding box satu-lawan-satu dengan IoU ≥ 0,5.
- Mean Absolute Error (MAE) Counting berdasarkan selisih absolut jumlah kandidat prediksi dan jumlah anotasi per gambar.

### Perbandingan SVM dan Random Forest

| Metrik | SVM | Random Forest |
|---|---:|---:|
| Precision Deteksi | 0,420 | **0,423** |
| Recall Deteksi | **0,289** | 0,287 |
| F1-score Deteksi | **0,343** | 0,342 |
| MAE Counting | **5,917** | 6,083 |

SVM menghasilkan MAE Counting paling rendah, yaitu 5,917 buah per gambar, sedangkan Random Forest menghasilkan precision deteksi sedikit lebih tinggi.

Perbedaan hasil validation dan test terjadi karena keduanya mengukur aspek yang berbeda. Validation mengukur klasifikasi kategori IoU, sedangkan evaluasi test mengukur kesesuaian lokasi dan jumlah objek yang terdeteksi.

Grafik perbandingan kedua model disimpan pada `M2/results/mae_comparison_ml.png`.

Hasil evaluasi tersedia pada:
- `M2/results/ml_results.csv`
- `M2/results/ml_test_evaluation.csv`
- `M2/results/test_evaluation.csv`
- `M2/results/counting_details.csv`

## Temuan dan Keterbatasan

1. Random Forest menunjukkan hasil lebih tinggi pada klasifikasi kategori IoU di validation set.
2. SVM menghasilkan rata-rata kesalahan counting sedikit lebih rendah pada data test.
3. Recall deteksi kedua model masih rendah karena kandidat dari tahap segmentasi awal belum sepenuhnya sesuai dengan anotasi.
4. Model Machine Learning hanya menyaring kandidat yang telah dihasilkan segmentasi, sehingga tidak dapat memulihkan objek anggur yang tidak terdeteksi sejak awal.
5. Pelabelan training masih berbasis IoU otomatis. Verifikasi label dan perbaikan kualitas kandidat diperlukan sebelum pengembangan model final.

Hasil eksperimen ini merupakan **baseline awal Machine Learning**, sehingga belum dapat dianggap sebagai performa akhir sistem penghitungan anggur.

## Isi Folder M2/

| File atau Folder | Fungsi |
|---|---|
| `feature_extraction.py` | Ekstraksi fitur HSV, bentuk, dan LBP serta penyusunan dataset fitur |
| `check_candidates.py` | Visualisasi sampel kandidat berdasarkan label IoU |
| `prepare_dataset.py` | Menyiapkan gambar dan CSV untuk pemeriksaan kandidat |
| `train_ml.py` | Training SVM dan Random Forest serta evaluasi validation |
| `evaluate_ml.py` | Evaluasi deteksi dan counting menggunakan data test |
| `results/features.csv` | Dataset fitur numerik |
| `results/dataset_stats.csv` | Statistik dataset dan kandidat |
| `results/ml_results.csv` | Hasil evaluasi klasifikasi validation |
| `results/cm_svm.png`, `cm_rf.png` | Confusion matrix kedua model |
| `results/svm_model.pkl`, `rf_model.pkl` | Model ML hasil training |
| `results/ml_test_evaluation.csv` | Evaluasi test khusus SVM dan Random Forest |
| `results/test_evaluation.csv` | Evaluasi test lengkap termasuk baseline segmentasi |
| `results/counting_details.csv` | Hasil penghitungan dan error per gambar |
| `results/mae_comparison_ml.png` | Grafik perbandingan MAE kedua model ML |
| `results/review/` | Gambar dan tabel pemeriksaan kandidat |

## Cara Menjalankan

Kode M2 memerlukan pipeline segmentasi dari M1 dan dataset hasil ekspor P0.

Pada implementasi saat ini, folder `M2/` berada di dalam direktori kerja M1, dengan struktur:

```text
M1/
├── preprocess.py
├── grape_pipeline.py
├── yolo_variants/
│   └── P0/
│       ├── train/
│       ├── valid/
│       └── test/
└── M2/
    ├── feature_extraction.py
    ├── train_ml.py
    ├── evaluate_ml.py
    └── results/
```

Instal dependensi:

```powershell
pip install opencv-python numpy pandas matplotlib scikit-image scikit-learn joblib
```

Jalankan perintah dari direktori M1:

```powershell
# 1. Ekstraksi fitur
python M2/feature_extraction.py

# 2. Pemeriksaan kandidat (opsional)
python M2/check_candidates.py
python M2/prepare_dataset.py

# 3. Training dan evaluasi validation
python M2/train_ml.py

# 4. Evaluasi deteksi dan counting
python M2/evaluate_ml.py
```

File hasil akan tersimpan pada folder `M2/results/`.

**Catatan integrasi:** Struktur direktori dalam repository kelompok menggunakan folder M1 dan M2 yang terpisah. Path pada kode M2 perlu disesuaikan dengan struktur repository final sebelum seluruh pipeline dijalankan dari root CountVision.

# Bagian M3: Model DL dan Integrasi

[Diisi oleh M3: model, perbandingan raw vs preprocessed, hasil, cara menjalankan prototipe.]

---

## Catatan untuk seluruh tim

- Pembagian data di `M1/data/splits/split.csv` bersifat tetap (per scene). Jangan diacak ulang.
- Test dipakai hanya untuk evaluasi akhir.
- Foto, dataset, dan model berukuran besar tidak diunggah ke repositori; bagikan lewat Google Drive.
