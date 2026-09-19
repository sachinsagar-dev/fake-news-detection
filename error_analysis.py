import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from src.text_utils import combine_title_text, load_dataset

RANDOM_STATE = 42
DATASET_PATH = "data/WELFake_Dataset.csv"
MODEL_PATH = "artifacts/fake_news_pipeline.joblib"
OUTPUT_DIR = Path("artifacts/error_analysis")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\nLoading dataset...")
    df = load_dataset(DATASET_PATH)
    print(f"Dataset loaded: {len(df):,} articles")

    print("\nApplying the same preprocessing as train.py...")
    df["title"] = df["title"].fillna("").astype(str)
    df["text"] = df["text"].fillna("").astype(str)
    df["content"] = combine_title_text(df)
    df = df[df["content"].str.len() > 20]
    df = df.drop_duplicates(subset=["content"])
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)

    print(f"Rows after cleaning: {len(df):,}")

    X_train, X_test, y_train, y_test = train_test_split(
        df["content"],
        df["label"],
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=df["label"],
    )

    print(f"Test samples: {len(X_test):,}")

    print("\nLoading trained model...")
    model = joblib.load(MODEL_PATH)

    print("\nGenerating predictions...")
    predictions = model.predict(X_test)

    probabilities = None
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X_test)[:, 1]

    results = pd.DataFrame({
        "content": X_test.values,
        "actual_label": y_test.values,
        "predicted_label": predictions,
    })

    if probabilities is not None:
        results["fake_probability"] = probabilities
        results["real_probability"] = 1 - probabilities

    results["actual"] = results["actual_label"].map({0: "REAL", 1: "FAKE"})
    results["predicted"] = results["predicted_label"].map({0: "REAL", 1: "FAKE"})
    results["correct"] = results["actual_label"] == results["predicted_label"]
    results["error_type"] = "CORRECT"

    results.loc[
        (results["actual_label"] == 0) & (results["predicted_label"] == 1),
        "error_type",
    ] = "FALSE_POSITIVE"

    results.loc[
        (results["actual_label"] == 1) & (results["predicted_label"] == 0),
        "error_type",
    ] = "FALSE_NEGATIVE"

    false_positives = results[results["error_type"] == "FALSE_POSITIVE"].copy()
    false_negatives = results[results["error_type"] == "FALSE_NEGATIVE"].copy()
    all_misclassified = results[~results["correct"]].copy()

    false_positives.to_csv(OUTPUT_DIR / "false_positives.csv", index=False)
    false_negatives.to_csv(OUTPUT_DIR / "false_negatives.csv", index=False)
    all_misclassified.to_csv(OUTPUT_DIR / "all_misclassified.csv", index=False)

    summary = {
        "test_samples": int(len(results)),
        "correct_predictions": int(results["correct"].sum()),
        "misclassified": int((~results["correct"]).sum()),
        "false_positives": int(len(false_positives)),
        "false_negatives": int(len(false_negatives)),
    }

    if probabilities is not None:
        summary["average_fake_probability"] = float(
            results["fake_probability"].mean()
        )

    with open(OUTPUT_DIR / "error_summary.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print("\n" + "=" * 60)
    print("ERROR ANALYSIS")
    print("=" * 60)
    print(f"Test samples        : {summary['test_samples']:,}")
    print(f"Correct predictions : {summary['correct_predictions']:,}")
    print(f"Misclassified       : {summary['misclassified']:,}")
    print(f"False positives     : {summary['false_positives']:,}")
    print(f"False negatives     : {summary['false_negatives']:,}")

    print("\nGenerated files:")
    for file in sorted(OUTPUT_DIR.iterdir()):
        print(" -", file.name)

    print("\nError analysis complete.")


if __name__ == "__main__":
    main()
