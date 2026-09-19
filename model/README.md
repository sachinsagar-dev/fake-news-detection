# Deployment model

This directory contains the trained inference pipeline required by the Streamlit deployment.

Expected file:

`model/fake_news_pipeline.joblib`

Generate it locally with:

```bash
python train.py --dataset data/WELFake_Dataset.csv
mkdir -p model
cp artifacts/fake_news_pipeline.joblib model/fake_news_pipeline.joblib
```

The trained model is intentionally stored separately from the training artifacts so the Streamlit app can load it directly during deployment.
