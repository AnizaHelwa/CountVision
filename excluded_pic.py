import os, shutil
import pandas as pd

df = pd.read_csv("metadata.csv")
os.makedirs("data/excluded", exist_ok=True)

for fn in ["grape_035.jpg", "grape_036.jpg"]:
    shutil.move(f"data/raw/{fn}", f"data/excluded/{fn}")

df["status"] = "dipakai"
df.loc[df["filename"].isin(["grape_035.jpg", "grape_036.jpg"]), "status"] = "excluded"
df.loc[df["status"] == "excluded", "catatan"] = "grayscale, hasil edit foto teman"
df["sumber"] = ""
df["scene_id"] = ""
df.to_csv("metadata.csv", index=False)