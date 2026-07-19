from __future__ import annotations

import json
from pathlib import Path
import pandas as pd
import streamlit as st
from src.inference import load_sentiment_pipeline, predict_review

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "models" / "best_sentiment_pipeline.joblib"
METADATA_PATH = PROJECT_ROOT / "models" / "best_sentiment_pipeline.metadata.json"
METRICS_PATH = PROJECT_ROOT / "output" / "metrics" / "model_comparison.csv"
CLASS_FIGURE_PATH = PROJECT_ROOT / "output" / "figures" / "class_distribution.png"
THEMES_PATH = PROJECT_ROOT / "output" / "insights" / "complaint_themes.csv"


@st.cache_resource
def load_model():
    return load_sentiment_pipeline(MODEL_PATH)


@st.cache_data
def load_optional_outputs():
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8")) if METADATA_PATH.is_file() else {}
    metrics = pd.read_csv(METRICS_PATH) if METRICS_PATH.is_file() else pd.DataFrame()
    themes = pd.read_csv(THEMES_PATH) if THEMES_PATH.is_file() else pd.DataFrame()
    return metadata, metrics, themes


st.set_page_config(page_title="Gojek Review Insight", layout="wide")
st.title("Gojek Review Insight")
st.caption("Portfolio analysis of rating-derived sentiment in Indonesian Gojek reviews. Predictions are not authoritative judgments.")
try:
    model = load_model()
except FileNotFoundError:
    st.error("The model artifact is missing. Run python -m src.sentiment_pipeline from the project root, then restart this app.")
    st.stop()
except Exception as exc:
    st.error(f"The model artifact could not be loaded: {exc}")
    st.stop()

metadata, metrics_df, themes_df = load_optional_outputs()
left, right = st.columns([1.25, 1])
with left:
    st.subheader("Review prediction")
    review = st.text_area("Indonesian review text", height=170, placeholder="Contoh: Pembayaran sering gagal dan aplikasinya lambat.")
    if st.button("Predict sentiment", use_container_width=True):
        try:
            prediction = predict_review(review, model)
        except ValueError as exc:
            st.warning(str(exc))
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
        else:
            st.success(f"Predicted sentiment: **{str(prediction['label']).title()}**")
            if prediction["confidence"] is not None:
                st.metric("Prediction confidence", f"{prediction['confidence']:.2%}")
                probability_df = pd.DataFrame(list(prediction["probabilities"].items()), columns=["sentiment", "probability"])
                st.bar_chart(probability_df.set_index("sentiment"))

with right:
    st.subheader("Model evaluation")
    if metadata:
        st.write(f"Selected model: **{metadata.get('model', '-')}**")
        st.write(f"Preprocessing: **{metadata.get('preprocessing_variant', '-')}**")
        macro_f1 = metadata.get("metrics", {}).get("macro_f1")
        if macro_f1 is not None:
            st.metric("Test macro F1", f"{macro_f1:.3f}")
    if not metrics_df.empty:
        st.dataframe(metrics_df.sort_values("macro_f1", ascending=False), use_container_width=True, hide_index=True)
    else:
        st.info("Optional model-comparison output is unavailable.")

st.subheader("Dataset class distribution")
if CLASS_FIGURE_PATH.is_file():
    st.image(str(CLASS_FIGURE_PATH), use_container_width=True)
else:
    st.info("Class-distribution figure is unavailable.")

st.subheader("Explainable complaint-theme categorization")
st.caption("Categories use documented keyword rules. Reviews may match more than one category; this is not topic modeling.")
if not themes_df.empty:
    visible = themes_df[["theme", "review_count", "percentage_of_negative_reviews"]].copy()
    visible["percentage_of_negative_reviews"] = visible["percentage_of_negative_reviews"].map(lambda value: f"{value:.2f}%")
    st.dataframe(visible, use_container_width=True, hide_index=True)
else:
    st.info("Complaint-theme output is unavailable.")
