from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from src.preprocessing import IndonesianTextPreprocessor

RANDOM_STATE = 42
TEST_SIZE = 0.2
LABELS = ["negative", "neutral", "positive"]


@dataclass
class ExperimentResult:
    model_name: str
    preprocessing_variant: str
    pipeline: Pipeline
    metrics: dict[str, float]
    training_time_seconds: float
    prediction_time_seconds: float
    predictions: np.ndarray
    probabilities: np.ndarray | None


def create_shared_split(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    splitter = StratifiedGroupKFold(
        n_splits=round(1 / TEST_SIZE),
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    train_index, test_index = next(
        splitter.split(df, df["sentiment"], groups=df["split_group"])
    )
    train_groups = set(df.iloc[train_index]["split_group"])
    test_groups = set(df.iloc[test_index]["split_group"])
    if train_groups & test_groups:
        raise RuntimeError(
            "Grouped split failed: normalized text overlaps train and test."
        )
    return train_index, test_index


def build_pipeline(variant: str, model_name: str) -> Pipeline:
    models = {
        "Multinomial Naive Bayes": MultinomialNB(alpha=1.0),
        "Logistic Regression": LogisticRegression(
            max_iter=1500, random_state=RANDOM_STATE
        ),
        "Logistic Regression Balanced": LogisticRegression(
            max_iter=1500, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
    }
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}")
    return Pipeline([
        ("preprocessor", IndonesianTextPreprocessor(variant=variant)),
        ("tfidf", TfidfVectorizer(
            max_features=8000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            sublinear_tf=True,
        )),
        ("model", models[model_name]),
    ])


def calculate_metrics(
    y_true: pd.Series, y_pred: np.ndarray
) -> dict[str, float]:
    macro = precision_recall_fscore_support(
        y_true, y_pred, labels=LABELS, average="macro", zero_division=0
    )
    weighted = precision_recall_fscore_support(
        y_true, y_pred, labels=LABELS, average="weighted", zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(macro[0]),
        "macro_recall": float(macro[1]),
        "macro_f1": float(macro[2]),
        "weighted_precision": float(weighted[0]),
        "weighted_recall": float(weighted[1]),
        "weighted_f1": float(weighted[2]),
    }


def run_experiments(
    train_df: pd.DataFrame, test_df: pd.DataFrame
) -> list[ExperimentResult]:
    model_names = [
        "Multinomial Naive Bayes",
        "Logistic Regression",
        "Logistic Regression Balanced",
        "Random Forest",
    ]
    results: list[ExperimentResult] = []
    for variant in ["basic", "linguistic"]:
        for model_name in model_names:
            pipeline = build_pipeline(variant, model_name)
            started = time.perf_counter()
            pipeline.fit(train_df["review_text"], train_df["sentiment"])
            training_time = time.perf_counter() - started
            started = time.perf_counter()
            predictions = pipeline.predict(test_df["review_text"])
            prediction_time = time.perf_counter() - started
            probabilities = (
                pipeline.predict_proba(test_df["review_text"])
                if hasattr(pipeline, "predict_proba")
                else None
            )
            results.append(ExperimentResult(
                model_name=model_name,
                preprocessing_variant=variant,
                pipeline=pipeline,
                metrics=calculate_metrics(test_df["sentiment"], predictions),
                training_time_seconds=training_time,
                prediction_time_seconds=prediction_time,
                predictions=predictions,
                probabilities=probabilities,
            ))
    return sorted(
        results,
        key=lambda result: (
            result.metrics["macro_f1"],
            result.metrics["weighted_f1"],
            -result.prediction_time_seconds,
        ),
        reverse=True,
    )
