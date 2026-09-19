import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from src.text_utils import (
    combine_title_text,
    load_fake_true,
    load_dataset
)

RANDOM_STATE = 42


# =========================================================
# MODEL DEFINITIONS
# =========================================================

def build_models():

    common = dict(
        lowercase=False,
        max_features=60000,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True
    )

    models = {

        # -------------------------------------------------
        # Logistic Regression
        # -------------------------------------------------

        "logistic_regression": Pipeline([
            (
                "tfidf",
                TfidfVectorizer(**common)
            ),
            (
                "model",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced"
                )
            )
        ]),

        # -------------------------------------------------
        # Calibrated Linear SVM
        # -------------------------------------------------

        "linear_svm": Pipeline([
            (
                "tfidf",
                TfidfVectorizer(**common)
            ),
            (
                "model",
                CalibratedClassifierCV(
                    estimator=LinearSVC(
                        class_weight="balanced"
                    ),
                    cv=3
                )
            )
        ]),

        # -------------------------------------------------
        # Naive Bayes
        # -------------------------------------------------

        "naive_bayes": Pipeline([
            (
                "tfidf",
                TfidfVectorizer(**common)
            ),
            (
                "model",
                MultinomialNB()
            )
        ])
    }

    return models


# =========================================================
# MAIN
# =========================================================

def main():

    # -----------------------------------------------------
    # COMMAND-LINE ARGUMENTS
    # -----------------------------------------------------

    parser = argparse.ArgumentParser(
        description="Train Fake News Detection Models"
    )

    parser.add_argument(
        "--dataset",
        default="data/WELFake_Dataset.csv",
        help="Path to WELFake CSV dataset"
    )

    parser.add_argument(
        "--fake",
        default="",
        help="Path to Fake.csv"
    )

    parser.add_argument(
        "--true",
        default="",
        help="Path to True.csv"
    )

    parser.add_argument(
        "--output",
        default="artifacts",
        help="Directory for model outputs"
    )

    args = parser.parse_args()


    # -----------------------------------------------------
    # OUTPUT DIRECTORY
    # -----------------------------------------------------

    out = Path(args.output)

    out.mkdir(
        parents=True,
        exist_ok=True
    )


    # =====================================================
    # LOAD DATASET
    # =====================================================

    print("\nLoading dataset...")

    if args.fake and args.true:

        df = load_fake_true(
            args.fake,
            args.true
        )

    else:

        df = load_dataset(
            args.dataset
        )


    print(
        f"Dataset loaded: {len(df):,} articles"
    )


    # =====================================================
    # BASIC CLEANING
    # =====================================================

    print("\nCleaning dataset...")

    df["title"] = (
        df["title"]
        .fillna("")
        .astype(str)
    )

    df["text"] = (
        df["text"]
        .fillna("")
        .astype(str)
    )


    # -----------------------------------------------------
    # Combine title + article
    # -----------------------------------------------------

    df["content"] = combine_title_text(df)


    # -----------------------------------------------------
    # Remove extremely short articles
    # -----------------------------------------------------

    df = df[
        df["content"].str.len() > 20
    ]


    # -----------------------------------------------------
    # Remove duplicate articles
    # -----------------------------------------------------

    df = df.drop_duplicates(
        subset=["content"]
    )


    # -----------------------------------------------------
    # Ensure labels are integers
    # -----------------------------------------------------

    df["label"] = pd.to_numeric(
        df["label"],
        errors="coerce"
    )


    # Remove invalid labels

    df = df.dropna(
        subset=["label"]
    )


    df["label"] = (
        df["label"]
        .astype(int)
    )


    # =====================================================
    # LABEL VALIDATION
    # =====================================================

    # Labels have already been standardized
    # inside src/text_utils.py:
    #
    # 0 = REAL
    # 1 = FAKE

    if not set(
        df["label"].unique()
    ).issubset({0, 1}):

        raise ValueError(
            "Labels must contain only 0 and 1."
        )


    # =====================================================
    # DATASET SUMMARY
    # =====================================================

    real_count = int(
        (df["label"] == 0).sum()
    )

    fake_count = int(
        (df["label"] == 1).sum()
    )


    print(
        f"\nReal articles : {real_count:,}"
    )

    print(
        f"Fake articles : {fake_count:,}"
    )

    print(
        f"Total articles: {len(df):,}"
    )


    # =====================================================
    # TRAIN / TEST SPLIT
    # =====================================================

    print("\nCreating train/test split...")

    X_train, X_test, y_train, y_test = train_test_split(

        df["content"],

        df["label"],

        test_size=0.20,

        random_state=RANDOM_STATE,

        stratify=df["label"]
    )


    print(
        f"Training samples: {len(X_train):,}"
    )

    print(
        f"Testing samples : {len(X_test):,}"
    )


    # =====================================================
    # TRAIN MODELS
    # =====================================================

    metrics = []

    trained_models = {}


    models = build_models()


    for name, model in models.items():

        print(
            f"\nTraining {name}..."
        )


        # -------------------------------------------------
        # Train
        # -------------------------------------------------

        model.fit(
            X_train,
            y_train
        )


        # -------------------------------------------------
        # Prediction
        # -------------------------------------------------

        predictions = model.predict(
            X_test
        )


        # =================================================
        # BASIC METRICS
        # =================================================

        accuracy = accuracy_score(
            y_test,
            predictions
        )

        precision = precision_score(
            y_test,
            predictions,
            zero_division=0
        )

        recall = recall_score(
            y_test,
            predictions,
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            predictions,
            zero_division=0
        )


        # =================================================
        # ROC-AUC
        # =================================================

        roc_auc = None


        if hasattr(
            model,
            "predict_proba"
        ):

            probabilities = model.predict_proba(
                X_test
            )[:, 1]

            roc_auc = roc_auc_score(
                y_test,
                probabilities
            )


        elif hasattr(
            model,
            "decision_function"
        ):

            scores = model.decision_function(
                X_test
            )

            roc_auc = roc_auc_score(
                y_test,
                scores
            )


        # =================================================
        # STORE METRICS
        # =================================================

        metrics.append({

            "model": name,

            "accuracy": accuracy,

            "precision": precision,

            "recall": recall,

            "f1": f1,

            "roc_auc": roc_auc

        })


        trained_models[name] = model


        # =================================================
        # CLASSIFICATION REPORT
        # =================================================

        report = classification_report(

            y_test,

            predictions,

            target_names=[
                "REAL",
                "FAKE"
            ],

            zero_division=0
        )


        report_path = (
            out /
            f"{name}_classification_report.txt"
        )


        with open(
            report_path,
            "w",
            encoding="utf-8"
        ) as file:

            file.write(report)


    # =====================================================
    # MODEL COMPARISON
    # =====================================================

    results = pd.DataFrame(
        metrics
    )


    results = results.sort_values(
        by="f1",
        ascending=False
    )


    results.to_csv(
        out / "model_comparison.csv",
        index=False
    )


    # =====================================================
    # SELECT BEST MODEL
    # =====================================================

    best_name = results.iloc[0]["model"]

    best_model = trained_models[
        best_name
    ]


    print("\n" + "=" * 60)

    print("MODEL COMPARISON")

    print("=" * 60)

    print(
        results.to_string(
            index=False
        )
    )


    print(
        f"\nBest model: {best_name}"
    )


    # =====================================================
    # SAVE BEST MODEL
    # =====================================================

    model_path = (
        out /
        "fake_news_pipeline.joblib"
    )


    joblib.dump(
        best_model,
        model_path
    )


    print(
        f"\nSaved best pipeline: {model_path}"
    )


    # =====================================================
    # CONFUSION MATRIX
    # =====================================================

    best_predictions = best_model.predict(
        X_test
    )


    cm = confusion_matrix(
        y_test,
        best_predictions
    )


    plt.figure(
        figsize=(7, 6)
    )


    sns.heatmap(

        cm,

        annot=True,

        fmt="d",

        cmap="Blues",

        xticklabels=[
            "REAL",
            "FAKE"
        ],

        yticklabels=[
            "REAL",
            "FAKE"
        ]

    )


    plt.title(
        f"Confusion Matrix — {best_name}"
    )

    plt.xlabel(
        "Predicted Label"
    )

    plt.ylabel(
        "Actual Label"
    )


    plt.tight_layout()


    plt.savefig(
        out / "confusion_matrix.png",
        dpi=160
    )


    plt.close()


    # =====================================================
    # TRAINING SUMMARY
    # =====================================================

    summary = {

        "dataset_rows_after_cleaning":
            int(len(df)),

        "real_articles":
            real_count,

        "fake_articles":
            fake_count,

        "train_rows":
            int(len(X_train)),

        "test_rows":
            int(len(X_test)),

        "best_model":
            best_name,

        "best_accuracy":
            float(
                results.iloc[0]["accuracy"]
            ),

        "best_precision":
            float(
                results.iloc[0]["precision"]
            ),

        "best_recall":
            float(
                results.iloc[0]["recall"]
            ),

        "best_f1":
            float(
                results.iloc[0]["f1"]
            ),

        "best_roc_auc":
            float(
                results.iloc[0]["roc_auc"]
            )
            if pd.notna(
                results.iloc[0]["roc_auc"]
            )
            else None
    }


    summary_path = (
        out /
        "training_summary.json"
    )


    summary_path.write_text(

        json.dumps(
            summary,
            indent=2
        ),

        encoding="utf-8"
    )


    # =====================================================
    # FINAL MESSAGE
    # =====================================================

    print("\n" + "=" * 60)

    print("TRAINING COMPLETE")

    print("=" * 60)

    print(
        f"Best model : {best_name}"
    )

    print(
        f"F1 score   : {summary['best_f1']:.4f}"
    )

    print(
        f"Accuracy   : {summary['best_accuracy']:.4f}"
    )

    print(
        f"ROC-AUC    : {summary['best_roc_auc']:.4f}"
    )

    print(
        f"\nModel saved to:"
        f"\n{model_path}"
    )


# =========================================================
# PROGRAM ENTRY POINT
# =========================================================

if __name__ == "__main__":
    main()