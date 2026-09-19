# NewsGuard — Fake News Detection System

An end-to-end NLP and machine-learning project that screens English news articles as **likely real** or **likely fake**.

> Important: the classifier is a screening model. It does not independently verify facts, sources, or claims.

## Project scope

This repository is deliberately structured as a full Data Science workflow:

1. Problem definition
2. Dataset ingestion
3. Data cleaning and duplicate removal
4. Exploratory Data Analysis
5. NLP preprocessing
6. TF-IDF feature engineering
7. Model experimentation
8. Evaluation with accuracy, precision, recall, F1 and ROC-AUC
9. Confusion matrix
10. Model persistence
11. Streamlit deployment
12. Limitations and responsible interpretation

## Models

- Logistic Regression
- Linear SVM
- Multinomial Naive Bayes

The training script compares all three and saves the highest-F1 pipeline.

## Dataset

The project accepts either:

### Option A — WELFake

Place `WELFake_Dataset.csv` in `data/`.

Expected columns:

```text
title,text,label
```

For the commonly distributed WELFake format, `0` represents real and `1` represents fake.

### Option B — Fake.csv + True.csv

Place:

```text
data/Fake.csv
data/True.csv
```

The training script automatically assigns:

```text
Fake.csv → 1
True.csv → 0
```

## Installation

```bash
python -m venv .venv
```

Windows:

```powershell
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Train

WELFake:

```bash
python train.py --dataset data/WELFake_Dataset.csv
```

Fake/True dataset:

```bash
python train.py --fake data/Fake.csv --true data/True.csv
```

This produces:

```text
artifacts/
├── fake_news_pipeline.joblib
├── model_comparison.csv
├── training_summary.json
├── confusion_matrix.png
├── *_classification_report.txt
```

## EDA

```bash
python eda.py --dataset data/WELFake_Dataset.csv
```

or:

```bash
python eda.py --fake data/Fake.csv --true data/True.csv
```

## Run the application

After training:

```bash
streamlit run app.py
```

Open the local URL shown by Streamlit.

## Suggested interview explanation

### Problem

"Online misinformation can spread quickly, so I built an NLP-based screening system that learns linguistic patterns from labelled news articles and classifies new articles as likely real or fake."

### Pipeline

```text
Raw Data
   ↓
Cleaning
   ↓
EDA
   ↓
Title + Body
   ↓
TF-IDF
   ↓
Multiple ML Models
   ↓
Evaluation
   ↓
Best Model
   ↓
Saved Pipeline
   ↓
Streamlit Application
```

### Why TF-IDF?

TF-IDF converts text into numerical features based on how important a word or n-gram is to a document relative to the corpus. It is fast, interpretable and a strong baseline for classical text classification.

### Why compare multiple models?

Different classifiers make different assumptions. Comparing them on the same held-out test set gives a more defensible model-selection process than choosing one model arbitrarily.

### What would you improve?

- Source-separated and time-separated validation
- Better duplicate/leakage controls
- Publisher/source features
- Transformer models such as BERT/RoBERTa
- External evidence retrieval
- Human-in-the-loop fact checking
- Monitoring for distribution drift

## Limitations

A high benchmark score does not mean the system can determine truth in the real world. Dataset artifacts, publisher/style leakage, topic overlap and changes in news language can affect generalization. Use the output as a screening signal, not as proof.

## License / dataset attribution

Check the license and attribution requirements of whichever public dataset you use before redistributing it with this repository.
