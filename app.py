from pathlib import Path

import joblib
import streamlit as st

MODEL_PATH = Path("artifacts/fake_news_pipeline.joblib")

st.set_page_config(
    page_title="NewsGuard — Fake News Detection",
    page_icon="📰",
    layout="centered",
)

st.markdown(
    """
    <style>
    .result-card {
        padding: 1rem 1.2rem;
        border-radius: 12px;
        margin: 0.8rem 0;
        border: 1px solid rgba(128,128,128,0.25);
    }
    .small-note {
        color: #777;
        font-size: 0.9rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("📰 NewsGuard")
st.caption(
    "NLP-based screening system for classifying news articles as likely real or fake."
)

if not MODEL_PATH.exists():
    st.warning("Model is not trained yet.")
    st.code(
        "python train.py --dataset data/WELFake_Dataset.csv\n"
        "# or\n"
        "python train.py --fake data/Fake.csv --true data/True.csv",
        language="bash",
    )
    st.stop()

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

st.subheader("Analyze an article")

title = st.text_input(
    "Headline",
    placeholder="Enter the article headline",
)

article = st.text_area(
    "Article text",
    height=260,
    placeholder="Paste the complete article text here...",
)

analyze = st.button(
    "🔎 Analyze News",
    type="primary",
    use_container_width=True,
)

if analyze:
    content = f"{title} {article}".strip()

    if len(content) < 30:
        st.error("Please enter a headline or article with more text.")
        st.stop()

    pred = int(model.predict([content])[0])

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba([content])[0]
        real_probability = float(probabilities[0])
        fake_probability = float(probabilities[1])
        confidence = fake_probability if pred == 1 else real_probability
    else:
        st.error("The loaded model does not provide calibrated probabilities.")
        st.stop()

    st.divider()
    st.subheader("Analysis result")

    if pred == 1:
        st.error("⚠️ Model classification: LIKELY FAKE")
    else:
        st.success("✅ Model classification: LIKELY REAL")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Real probability", f"{real_probability * 100:.1f}%")

    with col2:
        st.metric("Fake probability", f"{fake_probability * 100:.1f}%")

    st.progress(fake_probability, text="Fake probability")

    st.metric("Model confidence", f"{confidence * 100:.1f}%")

    with st.expander("What does this result mean?"):
        st.write(
            "The model estimates which class the article most closely resembles "
            "based on language patterns learned during training."
        )
        st.write(
            "A higher probability does not mean the article has been fact-checked. "
            "The system does not verify sources, claims, dates, or external evidence."
        )

    st.info(
        "This is a machine-learning screening tool, not a fact-checking service. "
        "A prediction does not establish whether a claim is actually true."
    )

st.divider()

st.subheader("How it works")

steps = [
    ("1", "Input", "Headline and article text are combined."),
    ("2", "Preprocessing", "Text is cleaned and normalized."),
    ("3", "TF-IDF", "Unigrams and bigrams are converted into numerical features."),
    ("4", "Classifier", "A calibrated Linear SVM predicts the class."),
    ("5", "Result", "The app displays class probabilities and the screening result."),
]

for number, name, description in steps:
    st.markdown(f"**{number}. {name}** — {description}")

st.caption(
    "Model trained on the WELFake dataset. Results depend on the training data "
    "and may not generalize to every type of news content."
)
