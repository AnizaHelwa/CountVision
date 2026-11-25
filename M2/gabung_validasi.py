from pathlib import Path
import pandas as pd

root = Path(__file__).resolve().parent.parent

folder_m2 = root / "M2" / "results"
file_gt = root / "M1" / "data" / "labels" / "gt_counts.csv"

pred = pd.read_csv(folder_m2 / "prediksi_validasi_ml.csv")
gt = pd.read_csv(file_gt)

# Samakan nama foto tanpa tambahan nama dari Roboflow
pred["id_foto"] = pred["filename"].str.extract(r"(grape_\d+)")
gt["id_foto"] = gt["filename"].str.extract(r"(grape_\d+)")

if pred["id_foto"].isna().any() or gt["id_foto"].isna().any():
    raise ValueError("Ada nama foto yang tidak sesuai format.")

if gt["id_foto"].duplicated().any():
    raise ValueError("Ada ID foto duplikat pada ground truth.")

hasil = pred.merge(
    gt[["id_foto", "jumlah_gt", "split_roboflow"]],
    on="id_foto",
    how="left",
    validate="many_to_one"
)

if hasil["jumlah_gt"].isna().any():
    raise ValueError("Ada gambar validasi tanpa ground truth.")

hasil["error_svm"] = abs(hasil["jumlah_gt"] - hasil["SVM"])
hasil["error_rf"] = abs(hasil["jumlah_gt"] - hasil["RF"])

hasil = hasil[[
    "id_foto", "filename", "split_roboflow",
    "jumlah_gt", "SVM", "RF", "error_svm", "error_rf"
]]

hasil.to_csv(
    folder_m2 / "prediksi_validasi_lengkap.csv",
    index=False
)

print("\n5 contoh validasi:")
print(hasil.head(5).to_string(index=False))

print("\nJumlah foto:", len(hasil))
print("Split ground truth:", hasil["split_roboflow"].unique())
print("\nFile berhasil disimpan.")
