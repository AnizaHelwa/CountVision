import pandas as pd
import numpy as np

SEED = 42
rng = np.random.default_rng(SEED)

df = pd.read_csv("metadata.csv")
df = df[df["status"] == "dipakai"].copy()
df["domain"] = np.where(df["posisi"].isin(["tandan", "tandan_lepas"]), "tandan", "lepas")

# satu baris per scene, domain diambil dari mayoritas foto di scene itu
scenes = (df.groupby("scene_id")
            .agg(domain=("domain", lambda s: s.mode()[0]), n_foto=("filename", "count"))
            .reset_index())

split_of = {}
for dom, g in scenes.groupby("domain"):
    ids = g["scene_id"].to_numpy()
    rng.shuffle(ids)
    n = len(ids)
    n_test = max(2, round(n * 0.15))
    n_val = max(2, round(n * 0.15))
    for s in ids[:n_test]:
        split_of[s] = "test"
    for s in ids[n_test:n_test + n_val]:
        split_of[s] = "val"
    for s in ids[n_test + n_val:]:
        split_of[s] = "train"

df["split"] = df["scene_id"].map(split_of)
df[["filename", "scene_id", "domain", "split"]].to_csv("data/splits/split.csv", index=False)

print(df.groupby(["split", "domain"]).agg(foto=("filename", "count"),
                                           scene=("scene_id", "nunique")))