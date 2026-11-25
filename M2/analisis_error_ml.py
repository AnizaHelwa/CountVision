from pathlib import Path
import pandas as pd

folder = Path(__file__).resolve().parent / "results"

df = pd.read_csv(folder / "progress3_per_foto.csv")

models = ["SVM", "Random Forest"]
ringkasan = []

for model in models:
    gt = df["ground_truth"]
    pred = df[model]

    under = (pred < gt).sum()
    over = (pred > gt).sum()
    tepat = (pred == gt).sum()

    mae = (gt - pred).abs().mean()

    ringkasan.append({
        "Model": model,
        "Undercounting": int(under),
        "Overcounting": int(over),
        "Jumlah Tepat": int(tepat),
        "Total Gambar": len(df),
        "MAE": round(mae, 3)
    })

hasil = pd.DataFrame(ringkasan)

print("\n=== ANALISIS ERROR M2 ===")
print(hasil.to_string(index=False))

detail = df[
    ["filename", "ground_truth", "SVM", "Random Forest"]
].copy()

for model in models:
    detail[f"error_{model}"] = (
        detail["ground_truth"] - detail[model]
    ).abs()

    detail[f"jenis_{model}"] = "Jumlah Tepat"

    detail.loc[
        detail[model] < detail["ground_truth"],
        f"jenis_{model}"
    ] = "Undercounting"

    detail.loc[
        detail[model] > detail["ground_truth"],
        f"jenis_{model}"
    ] = "Overcounting"

hasil.to_csv(
    folder / "progress3_analisis_error.csv",
    index=False
)

detail.to_csv(
    folder / "progress3_detail_error_ml.csv",
    index=False
)

print("\n5 gambar dengan error Random Forest terbesar:")
print(
    detail.sort_values(
        "error_Random Forest",
        ascending=False
    ).head(5).to_string(index=False)
)

print("\nSelesai! File analisis error berhasil disimpan.")
