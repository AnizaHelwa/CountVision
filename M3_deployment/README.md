# Paket M3 untuk website

Paket ini sudah memuat `app/models/best.pt`, metadata, hasil test, kode preprocessing dan adaptor inferensi. Foto validasi asli tidak tersedia di repository. Lengkapi dari dataset di laptop Ifka tanpa training ulang.

1. Ekstrak ZIP ini. Pindahkan folder `M3_web` ke folder proyek yang berisi `.venv`, `M3` dan `yolo_variants`.
2. Di terminal PowerShell pada folder proyek, jalankan:

```powershell
.\.venv\Scripts\python.exe .\M3_web\lengkapi_paket.py --dataset .\yolo_variants
```

3. Setelah berhasil, kirim `M3_web/M3_models_lengkap.zip` kepada teman website. ZIP tersebut berisi folder `app/models/` dengan lima foto validasi, CSV prediksi dan hasil benchmark CPU. Salin isinya ke proyek website sesuai struktur folder, setelah memeriksa model lama yang mungkin sudah ada.

Model default adalah P4. Website yang hanya menawarkan P0/P2/P3 perlu menambahkan P4 beserta preprocessing yang sudah disertakan. `app/models/README.md` berisi petunjuk pemanggilan fungsi integrasi. Jangan menggunakan foto validasi yang sudah P4 sebagai input mentah tanpa flag `already_p4=True`.

Paket belum di-push ke repository atau dipasang di website. Sintaks skrip sudah diperiksa; inferensi dan benchmark memerlukan environment serta foto lokal pengguna.
