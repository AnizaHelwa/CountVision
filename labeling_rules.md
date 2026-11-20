# Aturan Pelabelan Anggur

## Format
- Satu kelas: `grape`
- Anotasi berupa bounding box, diekspor ke format YOLO

## Aturan
1. Box dibuat rapat di sekeliling bagian buah yang terlihat. Tangkai tidak ikut.
2. Buah yang terlihat sekitar 50% atau lebih dilabel. Kurang dari itu tidak dilabel.
3. Buah yang terpotong tepi foto mengikuti aturan 50% yang sama.
4. Buah yang tertutup buah lain dilabel jika bagian terlihatnya memenuhi aturan 50%.
5. Bayangan dan pantulan tidak ikut di dalam box.

## Kasus yang diputuskan selama pelabelan
(isi di sini setiap kali menemukan kasus ambigu, supaya konsisten)

-