import pandas as pd

df = pd.read_csv("metadata.csv")
df = df[df["status"] == "dipakai"]   # abaikan foto yang di-exclude

print("Total foto dipakai:", len(df))
print("Jumlah scene unik :", df["scene_id"].nunique(), "\n")

for col in ["background", "pencahayaan", "posisi", "sudut", "jarak"]:
    print(df[col].value_counts(dropna=False), "\n")

# berapa scene independen per kategori (lebih jujur daripada jumlah foto)
for col in ["background", "pencahayaan", "posisi"]:
    print(f"Scene unik per {col}:")
    print(df.groupby(col)["scene_id"].nunique(), "\n")

# foto per scene
print("Foto per scene (terbanyak):")
print(df["scene_id"].value_counts().head(10))