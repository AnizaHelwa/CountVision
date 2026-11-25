from pathlib import Path
import cv2
import joblib
import numpy as np
import pandas as pd

root = Path(__file__).resolve().parent.parent
folder_hasil = root / "M2" / "results"
folder_output = folder_hasil / "visualisasi_progress3"
folder_output.mkdir(parents=True, exist_ok=True)

pilihan_folder = [
    root / "M1" / "yolo_variants" / "P0" / "test" / "images",
    root.parent / "M1" / "yolo_variants" / "P0" / "test" / "images",
    Path.home() / "Downloads" / "M1" / "yolo_variants" / "P0" / "test" / "images"
]

folder_gambar = next(
    (f for f in pilihan_folder if f.is_dir()),
    None
)

if folder_gambar is None:
    raise FileNotFoundError("Folder gambar test tidak ditemukan.")

folder_label = root / "M1" / "data" / "labels" / "yolo" / "test" / "labels"

data = pd.read_csv(folder_hasil / "features.csv")
data_test = data[data["split"] == "test"].copy()

kolom_bukan_fitur = [
    "filename", "split", "candidate_id",
    "x1", "y1", "x2", "y2", "label", "iou"
]

kolom_fitur = [k for k in data.columns if k not in kolom_bukan_fitur]

model_ml = {
    "SVM": joblib.load(folder_hasil / "svm_model.pkl"),
    "Random Forest": joblib.load(folder_hasil / "rf_model.pkl")
}


def baca_label(path, lebar, tinggi):
    kotak = []

    with open(path, "r") as file:
        for baris in file:
            if not baris.strip():
                continue

            kelas, xc, yc, w, h = map(float, baris.split())

            if int(kelas) != 0:
                continue

            x1 = int((xc - w / 2) * lebar)
            y1 = int((yc - h / 2) * tinggi)
            x2 = int((xc + w / 2) * lebar)
            y2 = int((yc + h / 2) * tinggi)

            kotak.append((x1, y1, x2, y2))

    return kotak


def gambar_kotak(img, kotak, warna):
    for x1, y1, x2, y2 in kotak:
        cv2.rectangle(
            img,
            (int(x1), int(y1)),
            (int(x2), int(y2)),
            warna,
            2
        )


def buat_panel(img, kotak_gt, kotak_pred, nama_model):
    hasil = img.copy()

    gambar_kotak(hasil, kotak_gt, (0, 255, 0))
    gambar_kotak(hasil, kotak_pred, (0, 0, 255))

    jumlah_gt = len(kotak_gt)
    jumlah_pred = len(kotak_pred)
    error = abs(jumlah_gt - jumlah_pred)

    tinggi, lebar = hasil.shape[:2]
    header = np.full((110, lebar, 3), 245, dtype=np.uint8)

    cv2.putText(
        header, nama_model, (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX, 0.9,
        (30, 30, 30), 2
    )

    cv2.putText(
        header,
        f"GT: {jumlah_gt} | Pred: {jumlah_pred} | Error: {error}",
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX, 0.65,
        (30, 30, 30), 2
    )

    return np.vstack((header, hasil))


def visualisasi(nama_foto):
    daftar_foto = list(folder_gambar.glob(f"{nama_foto}_*.jpg"))

    if len(daftar_foto) != 1:
        print(f"Gambar {nama_foto} tidak ditemukan atau duplikat.")
        return

    path_foto = daftar_foto[0]
    img = cv2.imread(str(path_foto))

    if img is None:
        print(f"Gagal membaca {path_foto.name}")
        return

    tinggi, lebar = img.shape[:2]

    path_label = folder_label / f"{path_foto.stem}.txt"

    if not path_label.exists():
        print(f"Label {nama_foto} tidak ditemukan.")
        return

    kotak_gt = baca_label(path_label, lebar, tinggi)

    kandidat = data_test[data_test["filename"] == path_foto.name].copy()

    if kandidat.empty:
        print(f"Tidak ada kandidat untuk {nama_foto}")
        return

    if kandidat["x2"].max() > lebar or kandidat["y2"].max() > tinggi:
        print(f"Koordinat kandidat tidak sesuai untuk {nama_foto}")
        return

    panel = []

    print(f"\nGambar: {nama_foto}")
    print(f"Ground truth: {len(kotak_gt)}")

    for nama_model, model in model_ml.items():
        prediksi = model.predict(kandidat[kolom_fitur])

        kandidat_lolos = kandidat[prediksi == 1]
        kotak_pred = kandidat_lolos[
            ["x1", "y1", "x2", "y2"]
        ].values.tolist()

        print(f"{nama_model}: {len(kotak_pred)} anggur")

        hasil = buat_panel(
            img, kotak_gt, kotak_pred, nama_model
        )

        panel.append(hasil)

    gabungan = np.hstack(panel)

    path_output = folder_output / f"{nama_foto}_perbandingan.png"
    cv2.imwrite(str(path_output), gabungan)

    print(f"Disimpan: {path_output}")


visualisasi("grape_006")
visualisasi("grape_102")

print("\nVisualisasi selesai.")
