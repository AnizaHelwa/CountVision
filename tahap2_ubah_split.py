import pandas as pd

sp = pd.read_csv("data/splits/split.csv")
sp.loc[sp["scene_id"] == "S_papan_01", "split"] = "test"
sp.to_csv("data/splits/split.csv", index=False)

print(sp.groupby(["split", "domain"]).agg(foto=("filename", "count"),
                                           scene=("scene_id", "nunique")))