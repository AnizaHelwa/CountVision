# Analisis Prediksi Seluruh Foto

## Cakupan

Seluruh 108 foto diproses menggunakan lima varian YOLO, yaitu
P0–P4, sehingga menghasilkan 540 gambar perbandingan anotasi
dan prediksi.

Setiap varian mencakup 61 foto train, 23 foto validasi, dan
24 foto test. Model tidak dilatih ulang. Confidence tetap
menggunakan hasil pemilihan pada validasi sebelumnya.

Hasil train, validasi, dan test dilaporkan secara terpisah.
Hasil train merupakan evaluasi pada data yang digunakan
selama pelatihan, bukan pengujian independen.

## Hasil P4

| Split | F1-score | MAE (butir/foto) | Foto dengan jumlah tepat |
|---|---:|---:|---:|
| Train | 97,97% | 0,328 | 45/61 |
| Validasi | 98,64% | 0,304 | 19/23 |
| Test | 99,19% | 0,125 | 21/24 |

P4 memperoleh F1 tertinggi dan MAE terendah pada ketiga split
dibandingkan varian lain dalam eksperimen ini.

Pada test, 21 dari 24 foto memiliki jumlah prediksi tepat.
Sebanyak 19 foto tidak memiliki FP maupun FN pada IoU 0,5.

## Analisis Kesalahan P4 pada Test

| Foto | Anotasi | Prediksi | TP | FP | FN |
|---|---:|---:|---:|---:|---:|
| grape_034 | 6 | 7 | 6 | 1 | 0 |
| grape_098 | 28 | 27 | 27 | 0 | 1 |
| grape_099 | 28 | 28 | 27 | 1 | 1 |
| grape_100 | 28 | 28 | 27 | 1 | 1 |
| grape_102 | 28 | 29 | 28 | 1 | 0 |

grape_034 dan grape_102 mengalami kelebihan hitung satu butir.
grape_098 mengalami kekurangan hitung satu butir.

Pada grape_099 dan grape_100, jumlah prediksi sesuai anotasi,
tetapi masing-masing memiliki satu FP dan satu FN. Kesalahan
tersebut saling menutupi dalam penghitungan jumlah.

FP dan FN dapat terjadi akibat objek yang terlewat, prediksi
tambahan, atau ketidaksesuaian lokasi dan ukuran kotak sehingga
tidak memenuhi IoU 0,5. Penyebab spesifik belum dapat dipastikan
tanpa pemeriksaan pasangan kotak yang tidak cocok.

## Kesimpulan

Jumlah prediksi yang tepat belum menjamin semua deteksi tepat.
Karena itu, MAE counting digunakan bersama F1-score deteksi.

Hasil ini berlaku pada dataset yang digunakan. Pengujian pada
foto baru dengan kondisi yang lebih beragam diperlukan untuk
menilai kemampuan generalisasi model.

## Lokasi Bukti

- Kode: prediksi_semua_foto.py
- Ringkasan: hasil_semua_foto/ringkasan_per_split.csv
- Gambar dan CSV per foto: hasil_semua_foto/P0 sampai P4,
  masing-masing dipisahkan menjadi train, valid, dan test.