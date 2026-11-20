import pandas as pd

d = pd.read_csv("hasil_v2_hsv/per_foto_B1_test.csv")
m = pd.read_csv("metadata.csv")[["filename", "scene_id"]]
d = d.merge(m, on="filename")

print(d.groupby("scene_id").agg(foto=("filename", "count"), gt=("n_gt", "mean"),
                                 pred=("n_pred", "mean"), MAE=("abs_err", "mean")).round(1))
print("\nFoto dengan kesalahan terbesar:")
print(d[["filename", "background", "pencahayaan", "n_gt", "n_pred", "abs_err"]]
      .sort_values("abs_err", ascending=False).head(8).to_string(index=False))