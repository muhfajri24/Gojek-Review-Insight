import joblib
import pytest
from src.inference import load_sentiment_pipeline, predict_review
from src.training import build_pipeline


@pytest.fixture
def fitted_pipeline():
    pipeline = build_pipeline("basic", "Multinomial Naive Bayes")
    texts = ["sangat buruk", "tidak bisa dipakai", "biasa saja", "cukup standar", "sangat bagus", "sangat membantu"]
    labels = ["negative", "negative", "neutral", "neutral", "positive", "positive"]
    pipeline.fit(texts, labels)
    return pipeline


def test_small_pipeline_predicts_one_sample(fitted_pipeline):
    assert fitted_pipeline.predict(["bagus sekali"])[0] in {"negative", "neutral", "positive"}


def test_saved_artifact_and_inference(tmp_path, fitted_pipeline):
    path = tmp_path / "model.joblib"
    joblib.dump(fitted_pipeline, path)
    result = predict_review("bagus sekali", load_sentiment_pipeline(path))
    assert result["label"] in {"negative", "neutral", "positive"}


def test_missing_artifact_has_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="Run python"):
        load_sentiment_pipeline(tmp_path / "missing.joblib")


def test_inference_rejects_empty_text(fitted_pipeline):
    with pytest.raises(ValueError, match="must not be empty"):
        predict_review("  ", fitted_pipeline)


def test_repository_artifact_can_be_loaded():
    pipeline = load_sentiment_pipeline()
    result = predict_review("aplikasi sangat membantu", pipeline)
    assert result["label"] in {"negative", "neutral", "positive"}
