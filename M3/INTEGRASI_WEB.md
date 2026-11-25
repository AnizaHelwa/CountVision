# Integrasi Website — M3 YOLO

## 1. Bobot Model

Lokasi bobot dari root repository:

`M3_deployment/app/models/best.pt`

Bobot tersebut merupakan salinan model P4:

`M3/models/P4_best.pt`

Ukuran file: 5.489.882 byte, sekitar 5,49 MB.

Untuk struktur website, salin isi folder
`M3_deployment/app/models/` ke `app/models/`.

## 2. Model dan Versi

- Arsitektur: YOLO11n, tugas object detection.
- Versi Ultralytics saat training: 8.4.213.
- Kelas: satu kelas, `grape`, indeks 0.
- Model dilatih selama 30 epoch.

## 3. Preprocessing

Model yang disiapkan untuk website menggunakan varian P4,
bukan P0, P2, atau P3. Website perlu mendukung preprocessing P4.

Urutan pemrosesan:

1. Resize dengan rasio aspek tetap, sisi terpanjang maksimal 1024 px.
2. Gamma correction: gamma 1,4.
3. CLAHE pada kanal L ruang warna LAB: clip 2,0, tile 8 × 8.
4. Bilateral filter: k 9, sigmaColor 50, sigmaSpace 50.
5. Sharpening: sigma 1,5, amount 0,7.
6. JPEG encode dengan quality 95, kemudian decode.

Model P4 tidak menggunakan mask segmentasi HSV sebagai input.
Eksperimen segmentasi merupakan model terpisah, yaitu P5_SEG.

Implementasi tersedia pada:

- `M3_deployment/app/models/preprocess_m1.py`
- `M3_deployment/app/models/inference.py`

## 4. Parameter Inferensi

| Parameter | Nilai |
|---|---:|
| imgsz | 640 |
| confidence | 0,50 |
| NMS IoU | 0,70 |
| max_det | 300 |
| device | cpu |

Confidence dipilih menggunakan data validasi.
NMS IoU 0,70 berbeda dari IoU pencocokan evaluasi, yaitu 0,50.

## 5. Fungsi Inferensi

Fungsi integrasi berada di
`M3_deployment/app/models/inference.py`.

`M3/demo.py` merupakan demo terpisah; gunakan fungsi integrasi
di atas untuk website.

Setelah folder model ditempatkan pada `app/models/`:

```python
import cv2
from app.models.inference import load_model, predict_bgr

model = load_model()

image = cv2.imread("foto.jpg")
result = predict_bgr(model, image)

jumlah_anggur = len(result.boxes)
gambar_hasil_bgr = result.plot()
```

Muat model sekali dan gunakan kembali untuk setiap prediksi.
Input fungsi harus berupa array gambar BGR.
Jika gambar berasal dari PIL/RGB, konversikan terlebih dahulu.

Untuk gambar yang sudah diproses P4:

```python
result = predict_bgr(model, image, already_p4=True)
```

Jangan melakukan preprocessing P4 dua kali.

## 6. Hasil Test

Evaluasi menggunakan 24 foto test dengan 429 anotasi.
Pencocokan prediksi dan anotasi dilakukan satu-ke-satu
dengan IoU minimal 0,50.

| Metrik | Nilai |
|---|---:|
| Precision | 99,0698% |
| Recall | 99,3007% |
| F1 deteksi | 99,1851% |
| MAE counting | 0,125 butir/foto |
| TP | 426 |
| FP | 4 |
| FN | 3 |
| Foto dengan hitungan tepat | 21 dari 24 |

Sumber hasil:

- `M3_deployment/app/models/test_metrics.json`
- `M3_deployment/app/models/evidence/P4_test_per_foto.csv`

Hasil tersebut khusus model P4, bukan P5_SEG.

## 7. Foto Validasi untuk Tes Integrasi

Folder:

`M3_deployment/app/models/validation_samples/`

Isi:

- Lima foto validasi yang sudah diproses P4.
- Label ground truth masing-masing foto.
- Gambar hasil prediksi.
- `predictions.csv` berisi nama foto, ground truth,
  predicted_count, dan absolute_count_error.

Gunakan kolom `predicted_count` sebagai hasil acuan integrasi,
bukan menggantinya dengan jumlah ground truth.

Contoh tes setelah folder model dipindahkan ke website:

```python
import csv
from pathlib import Path

import cv2
from app.models.inference import load_model, predict_bgr

folder = Path("app/models/validation_samples")
model = load_model()

with (folder / "predictions.csv").open(
    encoding="utf-8-sig", newline=""
) as file:
    rows = list(csv.DictReader(file))

for row in rows:
    image = cv2.imread(str(folder / row["filename"]))
    result = predict_bgr(model, image, already_p4=True)

    actual = len(result.boxes)
    expected = int(row["predicted_count"])

    assert actual == expected, (
        f"{row['filename']}: prediksi {actual}, acuan {expected}"
    )

print("Semua sampel validasi sesuai hasil acuan.")
```

Tes ini memeriksa pemuatan model dan kesesuaian jumlah prediksi
pada input yang sudah P4. Alur unggah gambar mentah,
preprocessing P4, dan tampilan website perlu diuji terpisah.

## 8. Benchmark CPU

Hasil benchmark tersedia pada:

`M3_deployment/app/models/cpu_benchmark.json`

- `mean_model_inference_ms`: rata-rata waktu inferensi model.
- `mean_predict_call_ms`: rata-rata waktu pemanggilan predict.

Benchmark menggunakan 3 warmup dan 25 pemanggilan terukur
pada lima foto validasi, batch 1.

Waktu tersebut tidak mencakup pembacaan file, preprocessing M1,
pemuatan model, rendering, atau jaringan.
Performa pada server dapat berbeda dari laptop pengujian.