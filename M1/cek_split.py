import pandas as pd

sp = pd.read_csv("data/splits/split.csv")
meta = pd.read_csv("metadata.csv")
df = meta.merge(sp[["filename", "domain", "split"]], on="filename")

for col in ["background", "pencahayaan", "sudut", "jarak"]:
    print(pd.crosstab(df[col], df["split"]), "\n")

print(df[df["domain"] == "lepas"].groupby("split")["scene_id"].apply(list))