# Model M3 untuk CountVision

Bobot `best.pt` berasal dari `M3/models/P4_best.pt`: YOLO11n, satu kelas `grape` (0), Ultralytics training 8.4.213. Ukuran 5.489.882 byte (5,49 MB). Nilai default: imgsz 640, confidence 0,50, NMS IoU 0,70, max_det 300, CPU.

## Kontrak integrasi

Model ini dilatih menggunakan P4. Tambahkan pilihan P4 di website dan gunakan P4 ketika memuat bobot ini. Jangan mengganti nama P4 menjadi P0/P2/P3 atau menganggap satu bobot mewakili semua varian. Jika website wajib hanya P0/P2/P3, gunakan bobot varian yang sesuai dan metrik varian tersebut.

Preprocessing input BGR: resize sisi terpanjang maksimal 1024, gamma 1,4 (LUT eksponen 1/gamma), CLAHE pada kanal L LAB (clip 2, tile 8), bilateral (k9, sigmaColor50, sigmaSpace50), sharpening (sigma1,5, amount0,7), JPEG quality95 lalu decode. Tidak memakai segmentasi HSV, mask, atau watershed pada input YOLO. Implementasi mengikuti demo M3 dan fungsi M1 yang disertakan.

```python
import cv2
from app.models.inference import load_model, predict_bgr

model = load_model()
image = cv2.imread("foto.jpg")
result = predict_bgr(model, image)
jumlah = len(result.boxes)
gambar_bgr = result.plot()
```

Muat model sekali saat aplikasi mulai/cache, bukan setiap request. Jika input berasal dari PIL/RGB, konversi ke BGR terlebih dahulu. Untuk foto contoh `validation_samples`, panggil `predict_bgr(model, image, already_p4=True)` karena sudah diproses P4; jangan preprocess dua kali. Saat menampilkan hasil ke komponen RGB, konversi BGR ke RGB.

## Bukti pengujian

Pada split test yang sama (24 foto, 429 anotasi), TP426 FP4 FN3, Precision 99,0698%, Recall 99,3007%, F1 99,1851%, MAE 0,125 butir/foto. Hitungan tepat 21/24 foto tidak berarti semua kotak deteksi benar. Pencocokan deteksi satu-ke-satu menggunakan IoU >=0,50. Confidence dipilih menggunakan 23 foto validasi. File CSV bukti disertakan di `evidence/`. Hasil ini diambil dari evaluasi proyek, bukan hasil menjalankan ulang model di server ini. Evaluasi 108 foto tetap dipisahkan menjadi train61/valid23/test24.

Waktu CPU historis: 160,4 ms inferensi model untuk satu gambar pada i5-13450HX. Ini bukan benchmark rata-rata. Jalankan skrip pelengkap untuk benchmark terukur pada komputer tujuan: 3 warmup, 5 foto validasi x5 pengulangan, batch1. `cpu_benchmark.json` mencatat waktu forward model dan waktu pemanggilan predict; tidak termasuk pembacaan file, preprocessing M1, pemuatan bobot, rendering dan jaringan.

## Foto validasi dan lingkungan

Jalankan `lengkapi_paket.py` dari paket induk untuk menambahkan lima foto validasi beserta anotasi, hasil kotak prediksi dan `predictions.csv`. Sampel dipilih berdasarkan posisi filename terurut (0,5,11,16,22), bukan berdasarkan hasil terbaik. Jumlah prediksi dihitung ulang dari bobot yang disertakan, tidak dikarang atau diberi label sebagai keluaran notebook.

Gunakan environment training Ifka yang sudah berjalan. Pin Ultralytics di `requirements-model.txt`; file ini bukan lockfile lengkap. Versi Python, torch dan CPU yang benar-benar dipakai benchmark akan dicatat skrip. Kompatibilitas server dan tampilan website belum diuji karena kode website belum tersedia dalam repo yang diperiksa.
