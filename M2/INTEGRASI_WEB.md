# Integrasi Model M2 untuk Website CountVision

## 1. Model yang digunakan

Model Machine Learning yang digunakan adalah SVM dan Random Forest.

File model:
- `M2/results/svm_model.pkl`
- `M2/results/rf_model.pkl`

SVM sudah berupa Pipeline yang berisi StandardScaler dan SVC. Random Forest menggunakan RandomForestClassifier tanpa scaler.

Kedua model menggunakan 21 fitur numerik hasil ekstraksi dari kandidat segmentasi.

Label prediksi:
- `1`: kandidat kategori positif berdasarkan IoU.
- `0`: kandidat kategori negatif berdasarkan IoU.

Model dapat dimuat menggunakan `joblib.load()` tanpa training ulang.

## 2. Ekstraksi fitur dan inferensi

Fungsi tersedia di `M2/inferensi_ml.py`.

Fungsi utama:
- `ekstrak_fitur(img, mask, box)` untuk mengekstraksi fitur kandidat.
- `load_model(nama_model)` untuk memuat model.
- `prediksi_ml(img, mask, boxes, model)` untuk menyeleksi kandidat dengan ML.

Input:
- `img`: citra BGR.
- `mask`: mask hasil segmentasi.
- `boxes`: daftar bounding box kandidat dengan format `(x1, y1, x2, y2)`.

Gambar, mask, dan koordinat bounding box harus menggunakan sistem ukuran yang sama dengan pipeline saat training.

Fitur yang digunakan:
- 6 fitur statistik HSV.
- 5 fitur geometri.
- 10 fitur histogram LBP.

Total: 21 fitur.

Ekstraksi fitur tidak cukup menggunakan `crop_bgr` saja karena fitur geometri membutuhkan mask dan ukuran gambar asal.

## 3. Pembentukan label kandidat

Label kandidat ditentukan berdasarkan nilai IoU maksimum terhadap bounding box ground truth.

- `1` jika IoU ≥ 0,5.
- `0` jika IoU ≤ 0,1.
- `-1` jika 0,1 < IoU < 0,5.

Kandidat berlabel `-1` tidak digunakan dalam training karena dianggap ambigu.

Label `0` tidak selalu berarti objek bukan anggur. Label tersebut menunjukkan bahwa kandidat memiliki kecocokan bounding box yang rendah terhadap anotasi.

## 4. Lingkungan Python yang diperiksa

- Python 3.13.15
- OpenCV 5.0.0
- NumPy 2.5.4
- Pandas 3.1.1
- Scikit-learn 1.9.1
- Scikit-image 0.26.0
- Joblib 1.6.0

Versi tersebut adalah versi lingkungan yang berhasil digunakan saat pemeriksaan model, bukan jaminan versi identik dengan lingkungan training awal.

## 5. Hasil evaluasi test

Pengujian menggunakan 24 gambar test dengan total 429 anotasi anggur. Pencocokan bounding box menggunakan IoU ≥ 0,5.

| Metode | Precision | Recall | F1 | MAE Counting |
|---|---:|---:|---:|---:|
| Segmentasi tanpa filter | 0,324 | 0,289 | 0,305 | 6,583 |
| Segmentasi + SVM | 0,420 | 0,289 | 0,343 | 5,917 |
| Segmentasi + Random Forest | 0,423 | 0,287 | 0,342 | 6,083 |

File evaluasi: `M2/results/test_evaluation.csv`.

## 6. Contoh validasi

| Foto | Ground Truth | SVM | Random Forest |
|---|---:|---:|---:|
| grape_008 | 5 | 5 | 5 |
| grape_018 | 11 | 10 | 10 |
| grape_022 | 11 | 11 | 11 |
| grape_025 | 8 | 12 | 12 |
| grape_029 | 8 | 8 | 8 |

Data lengkap 23 gambar validasi tersedia di `M2/results/prediksi_validasi_lengkap.csv`.

Foto asli tidak disertakan dalam paket ini dan perlu diambil dari dataset bersama.

## 7. Catatan integrasi

Model SVM dan Random Forest tidak menerima gambar mentah secara langsung.

Alur inferensi adalah:

Citra BGR → Preprocessing dan segmentasi M1 → Kandidat dan mask → Ekstraksi fitur M2 → Prediksi SVM/RF → Jumlah kandidat positif.

Jangan membuat scaler atau ekstraksi fitur baru yang berbeda dari pipeline training.

Fungsi integrasi sudah diperiksa dari sisi sintaks dan urutan fitur. Model juga menghasilkan jumlah prediksi positif yang konsisten pada data fitur test, tetapi integrasi ujung-ke-ujung dengan gambar baru dan pipeline M1 masih perlu diuji.

File model tetap berada di `M2/results/`. Jika website menggunakan folder `app/models/`, model dapat dimuat dari lokasi aslinya atau disalin ke folder tersebut saat integrasi.