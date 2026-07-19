from __future__ import annotations

from pathlib import Path
from typing import Any
import joblib
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "best_sentiment_pipeline.joblib"


def load_sentiment_pipeline(model_path: str | Path = DEFAULT_MODEL_PATH) -> Any:
    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Model artifact not found: {path}. Run python -m src.sentiment_pipeline to generate it."
        )
    return joblib.load(path)


def predict_review(review_text: str, pipeline: Any) -> dict[str, object]:
    if not review_text or not review_text.strip():
        raise ValueError("Review text must not be empty.")
    prediction = str(pipeline.predict([review_text])[0])
    result: dict[str, object] = {"label": prediction, "confidence": None, "probabilities": {}}
    if hasattr(pipeline, "predict_proba"):
        probabilities = pipeline.predict_proba([review_text])[0]
        labels = pipeline.named_steps["model"].classes_
        result["confidence"] = float(np.max(probabilities))
        result["probabilities"] = {str(label): float(probability) for label, probability in zip(labels, probabilities)}
    return result
