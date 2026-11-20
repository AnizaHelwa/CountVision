import pandas as pd

df = pd.read_csv("metadata.csv")

for col in ["background", "pencahayaan", "posisi", "sudut", "jarak",
            "sumber", "scene_id", "catatan"]:
    if col not in df.columns:
        df[col] = ""

# foto baru (hasil append) belum punya status
df["status"] = df["status"].fillna("dipakai")

df.to_csv("metadata.csv", index=False)
print(df.columns.tolist())