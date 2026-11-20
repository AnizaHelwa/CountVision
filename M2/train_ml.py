from pathlib import Path

import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    ConfusionMatrixDisplay
)

result_dir = Path(__file__).resolve().parent / "results"

df = pd.read_csv(result_dir / "features.csv")

# Hanya gunakan kandidat dengan label yang tidak ambigu
df = df[df["label"].isin([0, 1])].copy()

exclude = [
    "filename", "split", "candidate_id",
    "x1", "y1", "x2", "y2", "label", "iou"
]

features = [col for col in df.columns if col not in exclude]

train = df[df["split"] == "train"]
valid = df[df["split"] == "valid"]

X_train = train[features]
y_train = train["label"]

X_valid = valid[features]
y_valid = valid["label"]

print("Jumlah training:", len(train))
print("Jumlah validation:", len(valid))

svm = make_pipeline(
    StandardScaler(),
    SVC(kernel="rbf", C=1.0, class_weight="balanced")
)

rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced"
)

models = {
    "SVM": svm,
    "Random Forest": rf
}

results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_valid)

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_valid, pred),
        "Precision": precision_score(y_valid, pred, zero_division=0),
        "Recall": recall_score(y_valid, pred, zero_division=0),
        "F1-score": f1_score(y_valid, pred, zero_division=0)
    })

    filename = "svm_model.pkl" if name == "SVM" else "rf_model.pkl"
    joblib.dump(model, result_dir / filename)

    ConfusionMatrixDisplay.from_predictions(
        y_valid, pred,
        labels=[0, 1],
        display_labels=["IoU rendah", "IoU tinggi"],
        cmap="Blues"
    )

    plt.title(name)
    plt.tight_layout()

    image_name = "cm_svm.png" if name == "SVM" else "cm_rf.png"
    plt.savefig(result_dir / image_name, dpi=150)
    plt.close()

results_df = pd.DataFrame(results)
results_df.to_csv(
    result_dir / "ml_results.csv", index=False
)

print("\nHasil evaluasi validation:")
print(results_df.round(3).to_string(index=False))

print("\nModel dan hasil evaluasi berhasil disimpan.")
