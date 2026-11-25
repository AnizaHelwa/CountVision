from pathlib import Path
import pandas as pd

# Lokasi project
root = Path(__file__).resolve().parent.parent
m2_dir = root / "M2" / "results"
m3_dir = root / "M3" / "hasil_evaluasi_yolo"

# Membaca hasil evaluasi M2
ml = pd.read_csv(m2_dir / "counting_details.csv")
ml = ml[ml["model"].isin(["SVM", "Random Forest"])].copy()

# Pastikan tidak ada gambar yang tercatat dua kali per model
if ml.duplicated(["filename", "model"]).any():
    raise ValueError("Ada data M2 yang duplikat.")

# Ubah menjadi satu baris untuk setiap gambar
ml_table = ml.pivot(
    index="filename",
    columns="model",
    values="prediksi"
).reset_index()

# Ambil ground truth dari M2
gt_ml = (
    ml[ml["model"] == "SVM"][
        ["filename", "ground_truth"]
    ].rename(columns={"ground_truth": "gt_m2"})
)

ml_table = ml_table.merge(
    gt_ml, on="filename", validate="one_to_one"
)

# Baca hasil YOLO P0 sampai P4
hasil = ml_table.copy()

for varian in ["P0", "P1", "P2", "P3", "P4"]:
    path = m3_dir / f"{varian}_test_per_foto.csv"
    data = pd.read_csv(path)

    if data["filename"].duplicated().any():
        raise ValueError(f"Ada filename duplikat di {varian}")

    data = data[
        ["filename", "ground_truth", "prediksi"]
    ].rename(columns={
        "ground_truth": f"gt_{varian}",
        "prediksi": f"YOLO {varian}"
    })

    hasil = hasil.merge(
        data, on="filename",
        how="outer", validate="one_to_one",
        indicator=True
    )

    if not (hasil["_merge"] == "both").all():
        raise ValueError(
            f"Daftar gambar M2 dan YOLO {varian} tidak sama."
        )

    hasil = hasil.drop(columns="_merge")

# Pastikan semua ground truth sesuai
for varian in ["P0", "P1", "P2", "P3", "P4"]:
    if not (
        hasil["gt_m2"] == hasil[f"gt_{varian}"]
    ).all():
        raise ValueError(
            f"Ground truth M2 dan {varian} berbeda."
        )

# Pastikan jumlah gambar test sesuai
if len(hasil) != 24:
    raise ValueError(
        f"Seharusnya 24 gambar, ditemukan {len(hasil)}."
    )

hasil = hasil.rename(
    columns={"gt_m2": "ground_truth"}
)

hasil = hasil.drop(
    columns=[f"gt_{v}" for v in ["P0", "P1", "P2", "P3", "P4"]]
)

models = [
    "SVM", "Random Forest",
    "YOLO P0", "YOLO P1",
    "YOLO P2", "YOLO P3", "YOLO P4"
]

# Hitung error counting setiap model
for model in models:
    hasil[f"error_{model}"] = (
        hasil["ground_truth"] - hasil[model]
    ).abs()

# Kategori kesalahan khusus SVM
def jenis_error(row):
    if row["SVM"] < row["ground_truth"]:
        return "Undercounting"
    elif row["SVM"] > row["ground_truth"]:
        return "Overcounting"
    else:
        return "Jumlah sesuai"

hasil["jenis_error_SVM"] = hasil.apply(
    jenis_error, axis=1
)

# Simpan tabel per gambar
hasil = hasil.sort_values("filename")

hasil.to_csv(
    m2_dir / "progress3_per_foto.csv",
    index=False
)

# Membuat ringkasan counting
ringkasan = []

for model in models:
    ringkasan.append({
        "Model": model,
        "MAE Counting": hasil[f"error_{model}"].mean(),
        "Gambar Jumlah Tepat": int(
            (hasil[f"error_{model}"] == 0).sum()
        ),
        "Total Gambar": len(hasil)
    })

ringkasan_df = pd.DataFrame(ringkasan)

ringkasan_df.to_csv(
    m2_dir / "progress3_ringkasan.csv",
    index=False
)

# Menampilkan ringkasan
print("Jumlah gambar dibandingkan:", len(hasil))
print("Total anotasi anggur:", hasil["ground_truth"].sum())

print("\nRingkasan MAE Counting:")
print(ringkasan_df.round(3).to_string(index=False))

# Menampilkan gambar dengan error terbesar
print("\n5 gambar dengan kesalahan SVM terbesar:")

kolom = [
    "filename", "ground_truth",
    "SVM", "Random Forest", "YOLO P4",
    "error_SVM", "jenis_error_SVM"
]

print(
    hasil.sort_values(
        "error_SVM", ascending=False
    )[kolom].head(5).to_string(index=False)
)

print("\nBerhasil! Hasil disimpan di M2/results/")
