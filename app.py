from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = Path("model/fake_news_pipeline.joblib")

st.set_page_config(
    page_title="NewsGuard — Fake News Detection",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
    }
    .hero {
        padding: 1.2rem 1.4rem;
        border: 1px solid rgba(128,128,128,0.25);
        border-radius: 16px;
        background: linear-gradient(135deg, rgba(70,90,150,0.14), rgba(30,30,45,0.18));
        margin-bottom: 1.2rem;
    }
    .hero h1 {
        margin-bottom: 0.2rem;
    }
    .hero p {
        color: #8f96a3;
        margin-bottom: 0;
    }
    .section-card {
        padding: 1rem 1.1rem;
        border: 1px solid rgba(128,128,128,0.22);
        border-radius: 14px;
        min-height: 125px;
    }
    .section-card h4 {
        margin-top: 0;
        margin-bottom: 0.35rem;
    }
    .muted {
        color: #8f96a3;
        font-size: 0.9rem;
    }
    .result-card {
        padding: 1rem 1.2rem;
        border-radius: 12px;
        margin: 0.8rem 0;
        border: 1px solid rgba(128,128,128,0.25);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

if not MODEL_PATH.exists():
    st.error("Trained model is missing from the deployment package.")
    st.code(
        "Local setup: python train.py --dataset data/WELFake_Dataset.csv\n"
        "Then copy artifacts/fake_news_pipeline.joblib to model/fake_news_pipeline.joblib\n"
        "and commit the model for Streamlit deployment.",
        language="bash",
    )
    st.stop()


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


model = load_model()

# ---------------- Sidebar navigation ----------------
st.sidebar.title("📰 NewsGuard")
st.sidebar.caption("NLP-based fake-news screening")
page = st.sidebar.radio(
    "Navigation",
    ["Dashboard", "Analyze News", "Model Insights", "About"],
)

st.sidebar.divider()
st.sidebar.caption("Model: Calibrated Linear SVM")
st.sidebar.caption("Features: TF-IDF unigrams + bigrams")

# ---------------- Dashboard ----------------
if page == "Dashboard":
    st.markdown(
        """
        <div class="hero">
            <h1>📰 NewsGuard</h1>
            <p>Machine-learning screening for news articles based on textual patterns.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("Project overview")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Test accuracy", "97.06%")
    c2.metric("Test F1 score", "97.31%")
    c3.metric("ROC-AUC", "99.60%")
    c4.metric("Test articles", "12,646")

    st.divider()

    left, right = st.columns([1.35, 1])

    with left:
        st.subheader("Model comparison")
        comparison = pd.DataFrame(
            {
                "Model": [
                    "Calibrated Linear SVM",
                    "Logistic Regression",
                    "Multinomial Naive Bayes",
                ],
                "Accuracy": ["97.06%", "95.89%", "87.56%"],
                "F1": ["97.31%", "96.22%", "88.40%"],
                "ROC-AUC": ["99.60%", "99.26%", "94.33%"],
            }
        )
        st.dataframe(
            comparison,
            hide_index=True,
            use_container_width=True,
        )

    with right:
        st.subheader("Dataset snapshot")
        st.metric("After preprocessing", "63,227")
        d1, d2 = st.columns(2)
        d1.metric("Real", "28,708")
        d2.metric("Fake", "34,519")
        st.caption(
            "The benchmark uses a stratified 80/20 train-test split after "
            "removing very short and duplicate articles."
        )

    st.divider()
    st.subheader("How the system works")

    pipeline = [
        ("01", "Input", "Headline + article text"),
        ("02", "Clean", "Normalize and remove noise"),
        ("03", "TF-IDF", "Convert text into numerical features"),
        ("04", "SVM", "Classify learned language patterns"),
        ("05", "Result", "Display class probabilities"),
    ]

    cols = st.columns(5)
    for col, (number, name, description) in zip(cols, pipeline):
        with col:
            st.markdown(
                f"""
                <div class="section-card">
                    <div class="muted">{number}</div>
                    <h4>{name}</h4>
                    <div class="muted">{description}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.divider()
    st.subheader("Important limitation")
    st.info(
        "NewsGuard is a text-classification screening system, not a fact-checking "
        "service. It does not independently verify sources, claims, dates, or "
        "external evidence. Strong benchmark performance does not guarantee "
        "correct predictions on new or time-sensitive news."
    )

# ---------------- Analyze page ----------------
elif page == "Analyze News":
    st.markdown(
        """
        <div class="hero">
            <h1>🔎 Analyze News</h1>
            <p>Enter a headline and article body to run the trained classifier.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    input_col, help_col = st.columns([1.65, 0.75])

    with input_col:
        title = st.text_input(
            "Headline",
            placeholder="Enter the article headline",
        )
        article = st.text_area(
            "Article text",
            height=330,
            placeholder="Paste the complete article text here...",
        )

        analyze = st.button(
            "🔎 Analyze News",
            type="primary",
            use_container_width=True,
        )

    with help_col:
        st.subheader("Before you analyze")
        st.markdown(
            """
            **For best results**
            
            - Include the headline.
            - Paste enough of the article body.
            - Avoid submitting only a single sentence.
            - Treat the result as a screening signal.
            """
        )
        st.caption(
            "The model learned from WELFake and may behave differently on "
            "new topics, sources, or writing styles."
        )

    if analyze:
        content = f"{title} {article}".strip()

        if len(content) < 30:
            st.error("Please enter a headline or article with more text.")
            st.stop()

        pred = int(model.predict([content])[0])

        if not hasattr(model, "predict_proba"):
            st.error("The loaded model does not provide calibrated probabilities.")
            st.stop()

        probabilities = model.predict_proba([content])[0]
        real_probability = float(probabilities[0])
        fake_probability = float(probabilities[1])
        confidence = fake_probability if pred == 1 else real_probability

        st.divider()
        st.subheader("Analysis result")

        result_col, metrics_col = st.columns([1.1, 1])

        with result_col:
            if pred == 1:
                st.error("⚠️ Model classification: LIKELY FAKE")
            else:
                st.success("✅ Model classification: LIKELY REAL")

            st.progress(fake_probability, text="Fake-class probability")

        with metrics_col:
            m1, m2 = st.columns(2)
            m1.metric("Real probability", f"{real_probability * 100:.1f}%")
            m2.metric("Fake probability", f"{fake_probability * 100:.1f}%")
            st.metric("Model confidence", f"{confidence * 100:.1f}%")

        with st.expander("What does this result mean?"):
            st.write(
                "The model estimates which class the article most closely "
                "resembles based on language patterns learned during training."
            )
            st.write(
                "A higher probability does not mean the article has been "
                "fact-checked. The system does not verify sources, claims, "
                "dates, or external evidence."
            )

        st.info(
            "This is a machine-learning screening tool, not a fact-checking "
            "service. A prediction does not establish whether a claim is actually true."
        )

# ---------------- Model insights page ----------------
elif page == "Model Insights":
    st.markdown(
        """
        <div class="hero">
            <h1>📊 Model Insights</h1>
            <p>Evaluation details and error analysis from the held-out test set.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("Evaluation")
    e1, e2, e3 = st.columns(3)
    e1.metric("Test samples", "12,646")
    e2.metric("Correct predictions", "12,274")
    e3.metric("Misclassified", "372")

    st.divider()

    st.subheader("Confusion matrix summary")
    cm1, cm2, cm3, cm4 = st.columns(4)
    cm1.metric("True Real", "5,554")
    cm2.metric("False Positive", "188")
    cm3.metric("False Negative", "184")
    cm4.metric("True Fake", "6,720")

    st.divider()

    st.subheader("Why Linear SVM?")
    st.write(
        "The project compares Logistic Regression, a calibrated Linear SVM, "
        "and Multinomial Naive Bayes. The calibrated Linear SVM achieved the "
        "highest F1 score on the held-out test set and was selected for deployment."
    )

    st.subheader("What the error analysis showed")
    st.markdown(
        """
        - Some **real** articles use sensational or strongly opinionated language,
          making them resemble patterns associated with fake articles.
        - Some **fake** articles use conventional news-reporting language,
          making them resemble real articles.
        - The classifier learns correlations in text and labels; it does not
          independently establish the truth of a claim.
        """
    )

# ---------------- About page ----------------
else:
    st.markdown(
        """
        <div class="hero">
            <h1>ℹ️ About NewsGuard</h1>
            <p>An end-to-end NLP classification project built for learning, evaluation and deployment.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    a1, a2 = st.columns(2)

    with a1:
        st.subheader("Technology")
        st.markdown(
            """
            - Python
            - Pandas
            - Scikit-learn
            - TF-IDF
            - Logistic Regression
            - Linear SVM
            - Multinomial Naive Bayes
            - Streamlit
            - Joblib
            """
        )

    with a2:
        st.subheader("Project workflow")
        st.markdown(
            """
            **Dataset → preprocessing → EDA → TF-IDF → model comparison → 
            evaluation → error analysis → saved pipeline → Streamlit deployment**
            """
        )

    st.divider()
    st.subheader("Scope")
    st.write(
        "The system classifies articles according to patterns learned from the "
        "training dataset. It is intended as a screening demonstration and "
        "should not be used as an authoritative source of truth."
    )

    st.caption(
        "Training benchmark: WELFake. Results may not generalize to every type "
        "of news content, especially new or time-sensitive events."
    )
