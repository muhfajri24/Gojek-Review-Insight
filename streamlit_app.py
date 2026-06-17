from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from src.sentiment_pipeline import clean_text


PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "output" / "models" / "best_sentiment_model.joblib"
METRICS_PATH = PROJECT_ROOT / "output" / "reports" / "metrics_summary.csv"
SUMMARY_PATH = PROJECT_ROOT / "output" / "reports" / "business_insights.md"
PROJECT_SUMMARY_PATH = PROJECT_ROOT / "output" / "reports" / "project_summary.json"


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_metrics() -> pd.DataFrame:
    if not METRICS_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(METRICS_PATH)


@st.cache_data
def load_project_summary() -> dict[str, object]:
    if not PROJECT_SUMMARY_PATH.exists():
        return {}
    return json.loads(PROJECT_SUMMARY_PATH.read_text(encoding="utf-8"))


st.set_page_config(page_title="Gojek Review Insight", layout="wide")

st.title("Gojek Review Insight")
st.caption("A lightweight app for detecting sentiment in Indonesian-language Gojek user reviews.")

model = load_model()
metrics_df = load_metrics()
project_summary = load_project_summary()

if model is None:
    st.warning(
        "The model is not available yet. Run `python -m src.sentiment_pipeline` "
        "after placing the CSV file in `data/raw/gojek_reviews.csv`."
    )
    st.stop()

left, right = st.columns([1.4, 1])

with left:
    user_review = st.text_area(
        "Enter a user review",
        height=180,
        placeholder="Example: the Gojek app is very helpful, but lately it often fails during payment.",
    )

    if st.button("Predict Sentiment", use_container_width=True):
        if not user_review.strip():
            st.error("Please enter a review first.")
        else:
            cleaned_review = clean_text(user_review)
            prediction = model.predict([cleaned_review])[0]
            probabilities = model.predict_proba([cleaned_review])[0]
            labels = model.named_steps["model"].classes_
            confidence = float(probabilities[labels.tolist().index(prediction)])

            st.subheader("Prediction Result")
            st.success(f"Detected sentiment: **{prediction.title()}**")
            st.metric("Confidence Score", f"{confidence:.2%}")
            st.caption(f"Text after preprocessing: `{cleaned_review}`")

            probability_df = pd.DataFrame({"sentiment": labels, "probability": probabilities})
            st.bar_chart(probability_df.set_index("sentiment"))

with right:
    st.subheader("Model Summary")
    if project_summary:
        st.write(f"Best model: **{project_summary.get('best_model', '-')}**")
        st.write(f"Selection metric: **{project_summary.get('selection_metric', '-')}**")

    if not metrics_df.empty:
        st.dataframe(metrics_df, use_container_width=True)

    if SUMMARY_PATH.exists():
        st.subheader("Key Insights")
        st.markdown(SUMMARY_PATH.read_text(encoding="utf-8"))
