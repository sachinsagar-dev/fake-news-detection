# NewsGuard — Fake News Detection System

An end-to-end NLP and machine-learning project that screens English news articles as **likely real** or **likely fake**.

> **Important:** the classifier is a screening model. It does not independently verify facts, sources, or claims.

## Project scope

This repository is structured as a complete Data Science workflow:

1. Problem definition
2. Dataset ingestion
3. Data cleaning and duplicate removal
4. Exploratory Data Analysis
5. NLP preprocessing
6. TF-IDF feature engineering
7. Model experimentation
8. Evaluation with accuracy, precision, recall, F1 and ROC-AUC
9. Confusion matrix
10. Error analysis
11. Model persistence
12. Streamlit application
13. Limitations and responsible interpretation

## Project structure

```text
fake-news-detection/
├── app.py
├── train.py
├── eda.py
├── error_analysis.py
├── requirements.txt
├── PROJECT_PLAN.md
├── GIT_COMMANDS_LEARNING.md
├── src/
│   └── text_utils.py
├── data/
│   └── WELFake_Dataset.csv        # not included in Git
├── artifacts/
│   ├── fake_news_pipeline.joblib  # generated locally
│   ├── confusion_matrix.png
│   ├── model_comparison.csv
│   └── error_analysis/            # generated locally
└── notebooks/
    └── 01_fake_news_detection.ipynb
```

Generated datasets and ML artifacts are excluded through `.gitignore`. They can be recreated by following the dataset and training instructions below.

## Models

The training script compares:

- Logistic Regression
- Calibrated Linear SVM
- Multinomial Naive Bayes

The model with the highest F1 score is saved as the final pipeline.

## Dataset

The project accepts either of the following formats.

### Option A — WELFake

Place:

```text
data/WELFake_Dataset.csv
```

Expected columns:

```text
title,text,label
```

For the WELFake format used by this project, the raw dataset uses `0` = fake and `1` = real. The project normalizes these labels during loading to `0` = real and `1` = fake.

### Option B — Fake.csv + True.csv

Place:

```text
data/Fake.csv
data/True.csv
```

The training loader assigns:

```text
Fake.csv → 1 (FAKE)
True.csv → 0 (REAL)
```

> The dataset itself is not committed to this repository. Obtain it from its legitimate public source and follow its license/attribution requirements.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/sachinsagar-dev/fake-news-detection.git
cd fake-news-detection
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate the environment

Git Bash:

```bash
source .venv/Scripts/activate
```

Windows Command Prompt:

```cmd
.venv\Scripts\activate
```

PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

## Train the model

After placing the dataset in `data/`:

### WELFake

```bash
python train.py --dataset data/WELFake_Dataset.csv
```

### Fake.csv + True.csv

```bash
python train.py --fake data/Fake.csv --true data/True.csv
```

Training generates the model and evaluation artifacts under `artifacts/`.

## Exploratory Data Analysis

Run:

```bash
python eda.py --dataset data/WELFake_Dataset.csv
```

or:

```bash
python eda.py --fake data/Fake.csv --true data/True.csv
```

EDA generates class-distribution, article-length and word-frequency outputs under `artifacts/eda/`.

## Error analysis

After training:

```bash
python error_analysis.py
```

This recreates the same preprocessing and stratified test split used during training and generates:

```text
artifacts/error_analysis/
├── false_positives.csv
├── false_negatives.csv
├── all_misclassified.csv
└── error_summary.json
```

This helps inspect where the classifier makes mistakes rather than relying only on aggregate accuracy.

## Run the Streamlit application

After training:

```bash
streamlit run app.py
```

Open the local URL displayed by Streamlit.

The application displays:

- Likely real / likely fake classification
- Real probability
- Fake probability
- Model confidence
- Explanation of what the result means
- The model pipeline used for classification

## Reproducible workflow

A fresh setup follows this sequence:

```text
Clone repository
      ↓
Create virtual environment
      ↓
Install requirements.txt
      ↓
Obtain dataset
      ↓
Run train.py
      ↓
Run error_analysis.py (optional)
      ↓
Run streamlit run app.py
```

The trained model is generated locally by `train.py`; it is intentionally not required to be committed to Git.

## Results on the WELFake held-out test set

The current trained models produced:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Linear SVM | 97.06% | 97.28% | 97.33% | 97.31% | 99.60% |
| Logistic Regression | 95.89% | 96.61% | 95.83% | 96.22% | 99.26% |
| Naive Bayes | 87.56% | 90.06% | 86.79% | 88.40% | 94.33% |

The current best pipeline is the calibrated Linear SVM.

### Error analysis

On the same 12,646-sample held-out test set:

- Correct predictions: 12,274
- Misclassified: 372
- False positives: 188
- False negatives: 184

The error analysis showed that some real articles with strongly opinionated or sensational language can resemble fake articles, while some fake articles written in conventional news-reporting style can resemble real articles.

## Pipeline

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

## Why TF-IDF?

TF-IDF converts text into numerical features based on how important a word or n-gram is to a document relative to the corpus. It is fast, interpretable and a strong baseline for classical text classification.

This project uses unigrams and bigrams with sublinear TF scaling.

## Why compare multiple models?

Different classifiers make different assumptions. Comparing them on the same held-out test set provides a more defensible model-selection process than choosing one model arbitrarily.

## Limitations

A high benchmark score does not mean the system can determine truth in the real world.

The model:

- does not verify sources or claims;
- does not retrieve external evidence;
- can be affected by dataset-specific language patterns;
- can struggle with out-of-distribution articles;
- can produce false positives and false negatives;
- should not be treated as proof that an article is true or false.

In manual robustness probes, synthetic and paraphrased articles produced less decisive and sometimes incorrect classifications. These probes are qualitative demonstrations, not a formal external accuracy benchmark.

## Possible improvements

- Source-separated and time-separated validation
- Better duplicate/leakage controls
- Publisher/source features
- Transformer models such as BERT/RoBERTa
- External evidence retrieval
- Human-in-the-loop fact checking
- Monitoring for distribution drift

## Suggested interview explanation

### Problem

> "Online misinformation can spread quickly, so I built an NLP-based screening system that learns linguistic patterns from labelled news articles and classifies new articles as likely real or fake."

### End-to-end approach

> "I first loaded and normalized the dataset, cleaned the text and removed very short and duplicate records. I combined the headline and article body, converted the text into TF-IDF unigram and bigram features, and compared Logistic Regression, Linear SVM and Multinomial Naive Bayes. I selected the model using held-out test-set metrics, saved the best pipeline, performed error analysis on false positives and false negatives, and exposed the model through a Streamlit interface."

### Important limitation to mention

> "The model is a screening classifier, not a fact-checker. Its benchmark performance is strong on the WELFake test split, but real-world generalization can be affected by dataset artifacts and changes in news style."

## Git learning

A practical Git command guide based on the Git/GitHub issues encountered while building this project is available in:

```text
GIT_COMMANDS_LEARNING.md
```

It covers `git status`, `git diff`, `git fetch`, `git pull`, `git push`, fast-forward updates, untracked-file conflicts, `diff -u`, and rebase.

## License / dataset attribution

Check the license and attribution requirements of whichever public dataset you use before redistributing it with this repository.
