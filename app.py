from pathlib import Path
import joblib
import numpy as np
import streamlit as st

MODEL_PATH = Path("artifacts/fake_news_pipeline.joblib")

st.set_page_config(
    page_title="NewsGuard — Fake News Detection",
    page_icon="📰",
    layout="centered"
)

st.title("📰 NewsGuard")
st.caption("NLP-based screening system for classifying news articles as likely real or fake.")

if not MODEL_PATH.exists():
    st.warning("Model is not trained yet.")
    st.code(
        "python train.py --dataset data/WELFake_Dataset.csv\n"
        "# or\n"
        "python train.py --fake data/Fake.csv --true data/True.csv",
        language="bash"
    )
    st.stop()

model = joblib.load(MODEL_PATH)

title = st.text_input("Headline", placeholder="Enter the article headline")
article = st.text_area(
    "Article text",
    height=260,
    placeholder="Paste the complete article text here..."
)

if st.button("Analyze News", type="primary", use_container_width=True):
    content = f"{title} {article}".strip()

    if len(content) < 30:
        st.error("Please enter a headline or article with more text.")
        st.stop()

    pred = int(model.predict([content])[0])

    if hasattr(model, "predict_proba"):
        probability = float(model.predict_proba([content])[0][1])
        confidence = probability if pred == 1 else 1 - probability
    else:
        decision = float(model.decision_function([content])[0])
        confidence = 1 / (1 + np.exp(-abs(decision)))

    if pred == 1:
        st.error(f"⚠️ Model classification: LIKELY FAKE")
    else:
        st.success(f"✅ Model classification: LIKELY REAL")

    st.metric("Model confidence", f"{confidence * 100:.1f}%")

    st.info(
        "This is a machine-learning screening tool, not a fact-checking service. "
        "A prediction does not establish whether a claim is actually true."
    )

st.divider()
st.subheader("How it works")
st.markdown("""
1. Combines headline and article text.
2. Converts text into TF-IDF n-gram features.
3. Applies the trained classifier.
4. Returns a binary screening result and confidence estimate.
""")
