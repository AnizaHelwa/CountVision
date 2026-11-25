# Progress 3 — Evaluasi PCD + Machine Learning

## 1. Tujuan

Pada Progress 3, dilakukan evaluasi lanjutan terhadap integrasi metode Pengolahan Citra Digital (PCD) dengan Machine Learning menggunakan Support Vector Machine (SVM) dan Random Forest.

Kedua model digunakan untuk menyeleksi kandidat objek anggur yang dihasilkan melalui tahap preprocessing dan segmentasi sebelumnya.

## 2. Metode

Alur pengolahan terdiri dari:

1. Preprocessing dan segmentasi citra menggunakan pipeline M1.
2. Pembentukan kandidat objek anggur.
3. Ekstraksi fitur warna HSV, geometri, dan tekstur LBP.
4. Klasifikasi kandidat menggunakan SVM dan Random Forest.
5. Penghitungan kandidat yang lolos klasifikasi.
6. Evaluasi hasil terhadap anotasi ground truth.

Model SVM dan Random Forest yang telah dilatih pada Progress 2 digunakan kembali tanpa pelatihan ulang.

## 3. Hasil Evaluasi

Evaluasi dilakukan terhadap 24 gambar test dengan total 429 anotasi anggur. Pencocokan deteksi menggunakan IoU ≥ 0,5.

| Pipeline | Precision | Recall | F1-score | MAE Counting |
|---|---:|---:|---:|---:|
| Segmentasi tanpa filter ML | 0,324 | 0,289 | 0,305 | 6,583 |
| Segmentasi + SVM | 0,420 | 0,289 | 0,343 | 5,917 |
| Segmentasi + Random Forest | 0,423 | 0,287 | 0,342 | 6,083 |

Seluruh nilai dihitung menggunakan protokol evaluasi M2 yang sama.

## 4. Analisis Kesalahan Counting

| Jenis Hasil | SVM | Random Forest |
|---|---:|---:|
| Undercounting | 18 gambar | 17 gambar |
| Overcounting | 3 gambar | 3 gambar |
| Jumlah prediksi tepat | 3 gambar | 4 gambar |

Kedua model menunjukkan kecenderungan undercounting, terutama pada gambar dengan jumlah anggur yang banyak dan objek saling berhimpitan.

## 5. Contoh Keberhasilan dan Kegagalan

### grape_006 — Jumlah Prediksi Tepat

Ground truth: 10 anggur.

SVM: 10 prediksi.

Random Forest: 10 prediksi.

Pada gambar ini, anggur terpisah dengan cukup jelas sehingga kandidat objek lebih mudah dibentuk. Jumlah prediksi kedua model sesuai dengan ground truth. Namun, kesesuaian jumlah tidak selalu berarti seluruh bounding box identik dengan anotasi.

### grape_102 — Undercounting

Ground truth: 28 anggur.

Segmentasi tanpa filter ML: 12 prediksi.

SVM: 10 prediksi.

Random Forest: 10 prediksi.

Pada gambar ini, objek saling berhimpitan sehingga beberapa anggur kemungkinan tidak berhasil dipisahkan menjadi kandidat individual.

Berdasarkan hasil evaluasi, segmentasi tanpa filter menghasilkan TP 5 dan FP 7. Setelah filter SVM maupun Random Forest, TP tetap 5 dan FP turun menjadi 5.

Hal tersebut menunjukkan bahwa filter ML dapat mengurangi false positive, tetapi belum mampu mengatasi objek yang tidak berhasil dibentuk menjadi kandidat pada tahap sebelumnya.

## 6. Kesimpulan Sementara

Integrasi SVM dan Random Forest membantu meningkatkan precision serta mengurangi false positive dibandingkan kandidat segmentasi tanpa filter ML.

SVM menghasilkan MAE Counting terendah di antara dua metode ML, yaitu 5,917. Random Forest memiliki precision sedikit lebih tinggi, tetapi recall sedikit lebih rendah.

Keterbatasan utama pendekatan ini terletak pada ketergantungan terhadap kualitas kandidat hasil segmentasi. Model ML berperan sebagai filter dan tidak dapat menemukan kembali objek yang tidak terdeteksi pada tahap awal.

Evaluasi ini merupakan hasil sementara pada dataset test yang tersedia dan masih memerlukan pengujian lanjutan untuk menilai generalisasi metode.

## 7. File Hasil

- `analisis_progress3.py` — Perbandingan hasil counting ML dan YOLO.
- `analisis_error_ml.py` — Analisis jenis kesalahan counting.
- `visualisasi_ml.py` — Visualisasi bounding box SVM dan Random Forest.
- `results/progress3_per_foto.csv` — Tabel per gambar.
- `results/progress3_ringkasan.csv` — Ringkasan evaluasi counting.
- `results/progress3_analisis_error.csv` — Rekap jenis kesalahan.
- `results/progress3_detail_error_ml.csv` — Detail error per gambar.
- `results/visualisasi_progress3/` — Contoh visual keberhasilan dan kegagalan.