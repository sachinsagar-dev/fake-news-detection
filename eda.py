import argparse
from pathlib import Path
import re

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer


# ---------------------------------------------------------
# Arguments
# ---------------------------------------------------------

parser = argparse.ArgumentParser()

parser.add_argument("--dataset", default="")
parser.add_argument("--fake", default="")
parser.add_argument("--true", default="")
parser.add_argument("--output", default="artifacts/eda")

args = parser.parse_args()

out = Path(args.output)
out.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# Load dataset
# ---------------------------------------------------------

if args.fake and args.true:

    fake_df = pd.read_csv(args.fake)
    true_df = pd.read_csv(args.true)

    fake_df["label"] = 0
    true_df["label"] = 1

    df = pd.concat([fake_df, true_df], ignore_index=True)

else:

    df = pd.read_csv(args.dataset)


# ---------------------------------------------------------
# Basic cleaning
# ---------------------------------------------------------

df.columns = [str(c).strip().lower() for c in df.columns]

df["title"] = df["title"].fillna("").astype(str)
df["text"] = df["text"].fillna("").astype(str)

df["combined_text"] = df["title"] + " " + df["text"]

df["word_count"] = df["combined_text"].str.split().str.len()

# Original WELFake convention:
# 0 = FAKE
# 1 = REAL

df["label_name"] = df["label"].map({
    0: "Fake",
    1: "Real"
})


# ---------------------------------------------------------
# Basic dataset information
# ---------------------------------------------------------

print("=" * 60)
print("DATASET INFORMATION")
print("=" * 60)

print(f"Total articles: {len(df):,}")

print("\nColumns:")
print(df.columns.tolist())

print("\nMissing values:")
print(df[["title", "text", "label"]].isna().sum())


# ---------------------------------------------------------
# 1. CLASS DISTRIBUTION
# ---------------------------------------------------------

plt.figure(figsize=(7, 5))

sns.countplot(
    data=df,
    x="label_name"
)

plt.title("Fake vs Real News Distribution")
plt.xlabel("Article Type")
plt.ylabel("Number of Articles")

plt.tight_layout()

plt.savefig(
    out / "class_distribution.png",
    dpi=160
)

plt.close()


# ---------------------------------------------------------
# Class distribution numbers
# ---------------------------------------------------------

label_counts = df["label_name"].value_counts()

print("\n" + "=" * 60)
print("CLASS DISTRIBUTION")
print("=" * 60)

print(label_counts)

print("\nPercentage distribution:")

print(
    (label_counts / len(df) * 100)
    .round(2)
)


# ---------------------------------------------------------
# 2. ARTICLE LENGTH DISTRIBUTION
# ---------------------------------------------------------

sample_df = df.sample(
    min(10000, len(df)),
    random_state=42
)

plt.figure(figsize=(9, 5))

sns.histplot(
    data=sample_df,
    x="word_count",
    hue="label_name",
    bins=50,
    element="step"
)

plt.title("Article Length Distribution")
plt.xlabel("Number of Words")
plt.ylabel("Number of Articles")

plt.xlim(0, 3000)

plt.tight_layout()

plt.savefig(
    out / "article_length_distribution.png",
    dpi=160
)

plt.close()


# ---------------------------------------------------------
# 3. AVERAGE ARTICLE LENGTH
# ---------------------------------------------------------

length_summary = (
    df.groupby("label_name")["word_count"]
    .agg(["count", "mean", "median", "min", "max"])
    .round(2)
)

length_summary.to_csv(
    out / "text_length_summary.csv"
)


plt.figure(figsize=(7, 5))

sns.barplot(
    data=df,
    x="label_name",
    y="word_count",
    estimator="mean",
    errorbar=None
)

plt.title("Average Article Length")
plt.xlabel("Article Type")
plt.ylabel("Average Number of Words")

plt.tight_layout()

plt.savefig(
    out / "average_article_length.png",
    dpi=160
)

plt.close()


# ---------------------------------------------------------
# Print length statistics
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("ARTICLE LENGTH STATISTICS")
print("=" * 60)

print(length_summary)


# ---------------------------------------------------------
# 4. TOP WORDS
# ---------------------------------------------------------

def clean_for_words(text):

    text = text.lower()

    text = re.sub(
        r"https?://\S+|www\.\S+",
        " ",
        text
    )

    text = re.sub(
        r"[^a-z\s]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


print("\n" + "=" * 60)
print("TOP WORDS")
print("=" * 60)


# Limit text for memory efficiency
word_sample = df.sample(
    min(20000, len(df)),
    random_state=42
)


clean_text = word_sample["combined_text"].map(
    clean_for_words
)


vectorizer = CountVectorizer(
    stop_words="english",
    max_features=20
)


X = vectorizer.fit_transform(clean_text)


word_counts = X.sum(axis=0).A1

words = vectorizer.get_feature_names_out()


word_frequency = (
    pd.DataFrame({
        "word": words,
        "count": word_counts
    })
    .sort_values(
        "count",
        ascending=False
    )
)


print(word_frequency)


word_frequency.to_csv(
    out / "top_words.csv",
    index=False
)


# ---------------------------------------------------------
# 5. TOP WORDS VISUALIZATION
# ---------------------------------------------------------

plt.figure(figsize=(10, 6))

sns.barplot(
    data=word_frequency,
    x="count",
    y="word"
)

plt.title("Top 20 Most Frequent Words")
plt.xlabel("Frequency")
plt.ylabel("Word")

plt.tight_layout()

plt.savefig(
    out / "top_words.png",
    dpi=160
)

plt.close()


# ---------------------------------------------------------
# FINAL OUTPUT
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("EDA COMPLETE")
print("=" * 60)

print("\nGenerated files:")

for file in sorted(out.iterdir()):

    print(" -", file.name)

print("\nOutput directory:")
print(out)