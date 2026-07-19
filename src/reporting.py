from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report, confusion_matrix

from src.preprocessing import NEGATIONS, SLANG_MAP, STOPWORDS, basic_clean_text
from src.themes import categorize_complaint_themes
from src.training import ExperimentResult, LABELS, RANDOM_STATE

MIXED_SENTIMENT_TERMS = {"tapi", "namun", "padahal", "walaupun", "meskipun"}


def save_split_summary(
    df: pd.DataFrame,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    preparation_stats: dict[str, int],
    metrics_dir: Path,
) -> dict[str, object]:
    summary = {
        **preparation_stats,
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "class_distribution_total": df["sentiment"].value_counts().to_dict(),
        "class_distribution_train": train_df["sentiment"].value_counts().to_dict(),
        "class_distribution_test": test_df["sentiment"].value_counts().to_dict(),
        "normalized_text_overlap": len(
            set(train_df["split_group"]) & set(test_df["split_group"])
        ),
    }
    (metrics_dir / "dataset_split_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    rows = [
        {
            "split": split,
            "sentiment": label,
            "count": int(frame["sentiment"].eq(label).sum()),
        }
        for split, frame in [("total", df), ("train", train_df), ("test", test_df)]
        for label in LABELS
    ]
    pd.DataFrame(rows).to_csv(metrics_dir / "class_distribution.csv", index=False)
    return summary


def save_model_evaluation(
    results: list[ExperimentResult],
    best: ExperimentResult,
    test_df: pd.DataFrame,
    metrics_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    comparison = pd.DataFrame([
        {
            "model": result.model_name,
            "preprocessing_variant": result.preprocessing_variant,
            **result.metrics,
            "training_time_seconds": result.training_time_seconds,
            "prediction_time_seconds": result.prediction_time_seconds,
        }
        for result in results
    ])
    comparison.to_csv(metrics_dir / "model_comparison.csv", index=False)
    report = pd.DataFrame(classification_report(
        test_df["sentiment"],
        best.predictions,
        labels=LABELS,
        output_dict=True,
        zero_division=0,
    )).T.rename_axis("label").reset_index()
    report.to_csv(metrics_dir / "classification_report_best_model.csv", index=False)
    matrix = pd.DataFrame(
        confusion_matrix(
            test_df["sentiment"], best.predictions, labels=LABELS
        ),
        index=LABELS,
        columns=LABELS,
    )
    matrix.rename_axis("actual_label").reset_index().to_csv(
        metrics_dir / "confusion_matrix_best_model.csv", index=False
    )
    return comparison, report


def extract_frequent_terms(
    texts: Iterable[str], ngram_range: tuple[int, int], top_n: int = 20
) -> pd.DataFrame:
    cleaned = [basic_clean_text(text) for text in texts]
    cleaned = [text for text in cleaned if text]
    if not cleaned:
        return pd.DataFrame(columns=["term", "count"])
    vectorizer = CountVectorizer(
        binary=True,
        ngram_range=ngram_range,
        min_df=2,
        stop_words=sorted(STOPWORDS),
    )
    matrix = vectorizer.fit_transform(cleaned)
    counts = np.asarray(matrix.sum(axis=0)).ravel()
    order = counts.argsort()[::-1][:top_n]
    terms = vectorizer.get_feature_names_out()
    return pd.DataFrame({
        "term": terms[order],
        "count": counts[order].astype(int),
    })


def _save_figure(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def generate_figures(
    df: pd.DataFrame,
    comparison: pd.DataFrame,
    best: ExperimentResult,
    test_df: pd.DataFrame,
    figures_dir: Path,
) -> None:
    palette = {
        "negative": "#d97706", "neutral": "#94a3b8", "positive": "#2563eb"
    }
    counts = df["sentiment"].value_counts().reindex(LABELS)
    plt.figure(figsize=(8, 5))
    plt.bar(counts.index, counts.values, color=[palette[label] for label in counts.index])
    plt.title("Sentiment Class Distribution")
    plt.xlabel("Sentiment")
    plt.ylabel("Reviews")
    _save_figure(figures_dir / "class_distribution.png")

    plt.figure(figsize=(9, 5))
    sns.histplot(df["review_length"], bins=40, color="#2563eb")
    plt.title("Review Length Distribution")
    plt.xlabel("Characters per review")
    plt.ylabel("Reviews")
    _save_figure(figures_dir / "review_length_distribution.png")

    chart = comparison.sort_values("macro_f1")
    labels = chart["model"] + "\n(" + chart["preprocessing_variant"] + ")"
    plt.figure(figsize=(10, 6))
    plt.barh(labels, chart["macro_f1"], color="#2563eb")
    plt.xlim(0, 1)
    plt.title("Model Comparison by Macro F1")
    plt.xlabel("Macro F1")
    _save_figure(figures_dir / "model_comparison_macro_f1.png")

    matrix = confusion_matrix(
        test_df["sentiment"], best.predictions, labels=LABELS
    )
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        matrix, annot=True, fmt="d", cmap="Blues",
        xticklabels=LABELS, yticklabels=LABELS,
    )
    plt.title("Best Model Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    _save_figure(figures_dir / "confusion_matrix_best_model.png")

    for sentiment, file_name, color in [
        ("positive", "top_positive_terms.png", "#2563eb"),
        ("negative", "top_negative_terms.png", "#d97706"),
    ]:
        terms = extract_frequent_terms(
            df.loc[df["sentiment"] == sentiment, "review_text"], (1, 1)
        )
        plt.figure(figsize=(9, 6))
        plt.barh(terms["term"][::-1], terms["count"][::-1], color=color)
        plt.title(f"Top Terms in {sentiment.title()} Reviews")
        plt.xlabel("Review frequency")
        _save_figure(figures_dir / file_name)

    bigrams = extract_frequent_terms(
        df.loc[df["sentiment"] == "negative", "review_text"], (2, 2)
    )
    plt.figure(figsize=(10, 7))
    plt.barh(bigrams["term"][::-1], bigrams["count"][::-1], color="#d97706")
    plt.title("Top Bigrams in Negative Reviews")
    plt.xlabel("Review frequency")
    _save_figure(figures_dir / "top_negative_bigrams.png")


def export_feature_interpretation(
    best: ExperimentResult, insights_dir: Path, top_n: int = 25
) -> pd.DataFrame:
    model = best.pipeline.named_steps["model"]
    terms = best.pipeline.named_steps["tfidf"].get_feature_names_out()
    if hasattr(model, "coef_"):
        scores_by_class = model.coef_
    elif hasattr(model, "feature_log_prob_"):
        scores_by_class = model.feature_log_prob_
    else:
        output = pd.DataFrame(columns=["sentiment", "term", "score", "rank"])
        output.to_csv(insights_dir / "top_sentiment_terms.csv", index=False)
        return output
    rows: list[dict[str, object]] = []
    for class_index, sentiment in enumerate(model.classes_):
        order = np.argsort(scores_by_class[class_index])[::-1][:top_n]
        rows.extend({
            "sentiment": sentiment,
            "term": terms[index],
            "score": float(scores_by_class[class_index][index]),
            "rank": rank,
        } for rank, index in enumerate(order, 1))
    output = pd.DataFrame(rows)
    output.to_csv(insights_dir / "top_sentiment_terms.csv", index=False)
    return output


def export_error_analysis(
    best: ExperimentResult, test_df: pd.DataFrame, insights_dir: Path
) -> tuple[pd.DataFrame, str]:
    columns = ["review_text", "sentiment"]
    if "rating" in test_df:
        columns.append("rating")
    output = test_df[columns].copy().rename(columns={
        "sentiment": "actual_label", "rating": "rating_if_available"
    })
    output["predicted_label"] = best.predictions
    output["prediction_confidence"] = (
        best.probabilities.max(axis=1)
        if best.probabilities is not None else np.nan
    )
    output["error_type"] = (
        output["actual_label"] + " predicted " + output["predicted_label"]
    )
    cleaned = output["review_text"].map(basic_clean_text)
    output["flag_very_short"] = cleaned.str.split().str.len().le(2)
    output["flag_mixed_sentiment"] = cleaned.map(
        lambda value: bool(set(value.split()) & MIXED_SENTIMENT_TERMS)
    )
    output["flag_negation"] = cleaned.map(
        lambda value: bool(set(value.split()) & NEGATIONS)
    )
    output["flag_slang"] = output["review_text"].str.lower().map(
        lambda value: any(
            re.search(rf"\b{re.escape(term)}\b", value) for term in SLANG_MAP
        )
    )
    output["flag_repeated_characters"] = output["review_text"].map(
        lambda value: bool(re.search(r"(.)\1{2,}", value))
    )
    output["flag_emoji_heavy"] = output["review_text"].map(
        lambda value: len(re.findall(r"[^\x00-\x7F]", value)) >= 3
    )
    errors = output.loc[
        output["actual_label"] != output["predicted_label"]
    ].copy()
    errors.to_csv(insights_dir / "error_analysis.csv", index=False)
    flags = [column for column in errors if column.startswith("flag_")]
    lines = [
        "# Error Analysis Summary", "",
        "Automated flags are diagnostic heuristics, not human-validated causes.",
        "",
        f"- Test errors: {len(errors):,} of {len(output):,} "
        f"({len(errors) / len(output):.2%})",
    ]
    lines.extend(
        f"- {column.removeprefix('flag_').replace('_', ' ').title()}: "
        f"{int(errors[column].sum()):,} errors"
        for column in flags
    )
    lines.extend(["", "## Error types", ""])
    lines.extend(
        f"- {name}: {count:,}"
        for name, count in errors["error_type"].value_counts().items()
    )
    summary = "\n".join(lines) + "\n"
    (insights_dir / "error_analysis_summary.md").write_text(
        summary, encoding="utf-8"
    )
    return errors, summary


def write_business_summary(
    df: pd.DataFrame,
    comparison: pd.DataFrame,
    best: ExperimentResult,
    report: pd.DataFrame,
    themes: pd.DataFrame,
    insights_dir: Path,
) -> str:
    distribution = df["sentiment"].value_counts().reindex(LABELS).to_dict()
    hardest = report.loc[report["label"].isin(LABELS)].sort_values("f1-score").iloc[0]
    matched = themes.loc[themes["theme"] != "Other / uncategorized"]
    urgent = matched.sort_values("review_count", ascending=False).iloc[0]
    variants = comparison.groupby("preprocessing_variant")["macro_f1"].max().to_dict()
    summary = f"""# Business Summary

## Scope

- Reviews analyzed: {len(df):,}
- Sentiment distribution: {distribution}
- Labels are rating-derived: 1-2 negative, 3 neutral, and 4-5 positive.

## Model selection

- Selected model: {best.model_name} with {best.preprocessing_variant} preprocessing.
- Selection reason: highest test macro F1 ({best.metrics['macro_f1']:.4f}); weighted F1 was {best.metrics['weighted_f1']:.4f}.
- Best macro F1 by preprocessing experiment: {variants}.
- Hardest class by F1: {hardest['label']} ({hardest['f1-score']:.4f}).

## Evidence-backed finding

Finding:
The most frequent matched complaint category was **{urgent['theme']}**.

Evidence:
It matched {int(urgent['review_count']):,} negative reviews ({urgent['percentage_of_negative_reviews']:.2f}% of rating-derived negative reviews). Theme matches can overlap.

Recommendation:
Review representative examples and manually validate a sample before prioritizing product work. Use this category as a triage signal, not a causal diagnosis.

## Limitations

- Ratings are proxies for text sentiment and may disagree with review wording.
- Theme categorization uses transparent keyword rules, not topic modeling or human annotation.
- Theme categories overlap, so percentages do not sum to 100%.
- The default scope includes only app versions beginning with 4.8.
- Test results use one fixed grouped split and are not production-performance estimates.
"""
    (insights_dir / "business_summary.md").write_text(summary, encoding="utf-8")
    return summary


def save_model_and_metadata(
    best: ExperimentResult,
    dataset_path: Path,
    project_root: Path,
    models_dir: Path,
    split_summary: dict[str, object],
) -> dict[str, object]:
    artifact_path = models_dir / "best_sentiment_pipeline.joblib"
    joblib.dump(best.pipeline, artifact_path)
    metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_file": dataset_path.relative_to(project_root).as_posix(),
        "artifact_file": artifact_path.relative_to(project_root).as_posix(),
        "model": best.model_name,
        "preprocessing_variant": best.preprocessing_variant,
        "preprocessing_configuration": {
            "basic": "lowercase, URL and mention removal, hashtag content preservation, repeated-character normalization, punctuation cleanup, slang normalization",
            "linguistic": "basic plus negation-safe Indonesian stopwords and selective Sastrawi stemming for training tokens with frequency >= 5",
        },
        "selection_metric": "macro_f1",
        "metrics": best.metrics,
        "label_names": list(best.pipeline.named_steps["model"].classes_),
        "random_state": RANDOM_STATE,
        "split": split_summary,
        "label_rule": {"1-2": "negative", "3": "neutral", "4-5": "positive"},
        "training_scope": "appVersion starts with 4.8",
    }
    (models_dir / "best_sentiment_pipeline.metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return metadata
