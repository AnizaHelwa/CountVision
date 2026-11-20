import os, shutil
import pandas as pd

df = pd.read_csv("data/splits/split.csv")
for split in ["test", "val", "train"]:
    os.makedirs(f"upload_roboflow/{split}", exist_ok=True)

for _, r in df.iterrows():
    shutil.copy2(f"data/raw/{r['filename']}",
                 f"upload_roboflow/{r['split']}/{r['filename']}")

print(df["split"].value_counts())