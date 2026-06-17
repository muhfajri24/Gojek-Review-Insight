from __future__ import annotations

import json
import re
import textwrap
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

try:
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
    from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "Dependency 'Sastrawi' belum terpasang. Jalankan `pip install -r requirements.txt` "
        "di folder project, lalu coba lagi."
    ) from exc

try:
    from wordcloud import WordCloud
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "Dependency 'wordcloud' belum terpasang. Jalankan `pip install -r requirements.txt` "
        "di folder project, lalu coba lagi."
    ) from exc


sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
OUTPUT_DIR = PROJECT_ROOT / "output"
FIGURES_DIR = OUTPUT_DIR / "figures"
MODELS_DIR = OUTPUT_DIR / "models"
REPORTS_DIR = OUTPUT_DIR / "reports"

DEFAULT_CSV_NAME = "gojek_reviews.csv"
RANDOM_STATE = 42
TEST_SIZE = 0.2

TEXT_COLUMN_CANDIDATES = [
    "content",
    "review",
    "review_text",
    "text",
    "ulasan",
    "komentar",
    "comment",
    "review_body",
]
RATING_COLUMN_CANDIDATES = ["rating", "score", "star", "stars", "nilai"]
SENTIMENT_COLUMN_CANDIDATES = ["sentiment", "label", "polarity", "class", "kelas"]

CUSTOM_STOPWORDS = {
    "aplikasi",
    "app",
    "apk",
    "gojek",
    "nya",
    "yg",
    "aja",
    "bgt",
    "dong",
    "nih",
    "sih",
    "tau",
    "biar",
    "buat",
    "iya",
    "ga",
    "gak",
    "nggak",
    "udah",
    "kak",
    "min",
}


@dataclass
class ModelResult:
    name: str
    pipeline: Pipeline
    metrics: dict[str, float]
    confusion_matrix: np.ndarray
    y_pred: np.ndarray
    y_proba: np.ndarray


def ensure_directories() -> None:
    """Create the expected project directories if they do not exist."""
    for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, OUTPUT_DIR, FIGURES_DIR, MODELS_DIR, REPORTS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def find_dataset_path(csv_path: str | Path | None = None) -> Path:
    """Find the review CSV from the user-provided path or the raw-data folder."""
    if csv_path is not None:
        path = Path(csv_path)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        if path.exists():
            return path
        raise FileNotFoundError(f"CSV file was not found: {path}")

    preferred_path = RAW_DATA_DIR / DEFAULT_CSV_NAME
    if preferred_path.exists():
        return preferred_path

    csv_files = sorted(RAW_DATA_DIR.glob("*.csv"))
    if csv_files:
        return csv_files[0]

    raise FileNotFoundError(
        "No CSV dataset was found. Place the Kaggle CSV inside "
        f"'{RAW_DATA_DIR.relative_to(PROJECT_ROOT)}' and name it '{DEFAULT_CSV_NAME}', "
        "or pass an explicit csv_path."
    )


def load_dataset(csv_path: str | Path | None = None) -> pd.DataFrame:
    """Load the raw review dataset from CSV."""
    dataset_path = find_dataset_path(csv_path)
    return pd.read_csv(dataset_path)


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names for easier downstream processing."""
    normalized = df.copy()
    normalized.columns = [
        re.sub(r"[^a-z0-9]+", "_", str(column).strip().lower()).strip("_")
        for column in normalized.columns
    ]
    return normalized


def _find_first_matching_column(columns: Iterable[str], candidates: list[str]) -> str | None:
    column_set = {column.lower(): column for column in columns}
    for candidate in candidates:
        if candidate in column_set:
            return column_set[candidate]
    return None


def standardize_review_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Rename the key columns to review_text, rating, and sentiment when available."""
    normalized = normalize_column_names(df)
    text_column = _find_first_matching_column(normalized.columns, TEXT_COLUMN_CANDIDATES)
    rating_column = _find_first_matching_column(normalized.columns, RATING_COLUMN_CANDIDATES)
    sentiment_column = _find_first_matching_column(normalized.columns, SENTIMENT_COLUMN_CANDIDATES)

    if text_column is None:
        raise ValueError(
            "Review text column was not detected automatically. "
            "Expected one of: " + ", ".join(TEXT_COLUMN_CANDIDATES)
        )

    standardized = normalized.copy()
    rename_map = {text_column: "review_text"}
    if rating_column is not None:
        rename_map[rating_column] = "rating"
    if sentiment_column is not None:
        rename_map[sentiment_column] = "sentiment"
    standardized = standardized.rename(columns=rename_map)

    standardized["review_text"] = standardized["review_text"].fillna("").astype(str)
    if "rating" in standardized.columns:
        standardized["rating"] = pd.to_numeric(standardized["rating"], errors="coerce")
    return standardized


def filter_reference_app_version(df: pd.DataFrame, version_prefix: str = "4.8") -> pd.DataFrame:
    """Match the Kaggle reference notebook by focusing on app version 4.8 reviews when available."""
    if "appversion" not in df.columns:
        return df

    filtered = df.loc[df["appversion"].astype(str).str.startswith(version_prefix, na=False)].copy()
    return filtered if not filtered.empty else df


def normalize_sentiment_label(value: object) -> str | None:
    """Map raw labels into positive, negative, or neutral."""
    if pd.isna(value):
        return None

    text = str(value).strip().lower()
    mapping = {
        "positive": "positive",
        "positif": "positive",
        "pos": "positive",
        "1": "positive",
        "good": "positive",
        "negative": "negative",
        "negatif": "negative",
        "neg": "negative",
        "0": "negative",
        "-1": "negative",
        "bad": "negative",
        "neutral": "neutral",
        "netral": "neutral",
        "2": "neutral",
    }
    return mapping.get(text, text if text in {"positive", "negative", "neutral"} else None)


def infer_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """Create a sentiment label using the existing label or the numeric rating."""
    enriched = df.copy()

    if "sentiment" in enriched.columns:
        enriched["sentiment"] = enriched["sentiment"].apply(normalize_sentiment_label)

    if "rating" in enriched.columns:
        inferred = np.select(
            [
                enriched["rating"] >= 4,
                enriched["rating"] <= 2,
                enriched["rating"] == 3,
            ],
            ["positive", "negative", "neutral"],
            default=None,
        )
        if "sentiment" in enriched.columns:
            enriched["sentiment"] = enriched["sentiment"].fillna(pd.Series(inferred, index=enriched.index))
        else:
            enriched["sentiment"] = inferred

    if "sentiment" not in enriched.columns or enriched["sentiment"].dropna().empty:
        raise ValueError(
            "Sentiment labels could not be inferred. The dataset needs either a sentiment column "
            "or a rating column that can be mapped into sentiment."
        )

    return enriched


def get_stopwords() -> set[str]:
    """Combine Sastrawi stopwords with a small project-specific list."""
    stopword_factory = StopWordRemoverFactory()
    return set(stopword_factory.get_stop_words()) | CUSTOM_STOPWORDS


STEMMER = StemmerFactory().create_stemmer()
STOPWORDS = get_stopwords()


@lru_cache(maxsize=50000)
def stem_token_cached(token: str) -> str:
    """Cache token-level stemming so repeated Indonesian words are only stemmed once."""
    return STEMMER.stem(token)


def basic_clean_text(text: str) -> str:
    """Preprocess Indonesian review text before stemming."""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"@[A-Za-z0-9_]+", " ", text)
    text = re.sub(r"#[A-Za-z0-9_]+", " ", text)
    text = re.sub(r"[0-9]+", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"[\U00010000-\U0010ffff]", " ", text)
    text = re.sub(r"_+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    filtered_tokens = [token for token in text.split() if token not in STOPWORDS and len(token) > 1]
    return " ".join(filtered_tokens)


def clean_text(text: str, stem_vocabulary: set[str] | None = None) -> str:
    """Preprocess Indonesian review text with selective Sastrawi stemming for speed."""
    filtered_tokens = basic_clean_text(text).split()
    stemmed_tokens = [
        stem_token_cached(token) if stem_vocabulary is None or token in stem_vocabulary else token
        for token in filtered_tokens
    ]
    stemmed_tokens = [token for token in stemmed_tokens if token and token not in STOPWORDS and len(token) > 1]
    return " ".join(stemmed_tokens)


def preprocess_reviews(df: pd.DataFrame) -> pd.DataFrame:
    """Apply sentiment inference and text preprocessing to the raw dataset."""
    standardized = standardize_review_dataset(df)
    standardized = filter_reference_app_version(standardized)
    enriched = infer_sentiment(standardized)

    processed = enriched.copy()
    processed["review_text"] = processed["review_text"].fillna("").astype(str).str.strip()
    processed = processed.loc[processed["review_text"].ne("")].copy()
    processed = processed.drop_duplicates(subset=["review_text"]).reset_index(drop=True)

    processed["normalized_text"] = processed["review_text"].apply(basic_clean_text)
    token_frequency = (
        pd.Series(" ".join(processed["normalized_text"].fillna("")).split(), dtype="object").value_counts()
    )
    stem_vocabulary = set(token_frequency.loc[token_frequency >= 5].index.tolist())

    unique_reviews = processed["review_text"].unique().tolist()
    cleaned_lookup = {review: clean_text(review, stem_vocabulary) for review in unique_reviews}
    processed["clean_text"] = processed["review_text"].map(cleaned_lookup)
    processed["review_length"] = processed["review_text"].str.len()
    processed["token_count"] = processed["clean_text"].str.split().str.len().fillna(0).astype(int)

    processed = processed.loc[processed["clean_text"].str.strip().ne("")]
    processed = processed.reset_index(drop=True)

    processed.to_csv(PROCESSED_DATA_DIR / "gojek_reviews_processed.csv", index=False)
    return processed


def prepare_modeling_data(df: pd.DataFrame) -> pd.DataFrame:
    """Filter the dataset to positive and negative classes for supervised learning."""
    modeling_df = df.loc[df["sentiment"].isin(["positive", "negative"])].copy()
    if modeling_df["sentiment"].nunique() < 2:
        raise ValueError("At least two sentiment classes are required for model training.")
    return modeling_df.reset_index(drop=True)


def split_dataset(df: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Create a stratified train-test split using the cleaned text."""
    return train_test_split(
        df["clean_text"],
        df["sentiment"],
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df["sentiment"],
    )


def build_model_pipelines() -> dict[str, Pipeline]:
    """Build the candidate TF-IDF + classifier pipelines."""
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
    )

    return {
        "Naive Bayes": Pipeline(
            [
                ("tfidf", clone(vectorizer)),
                ("model", MultinomialNB()),
            ]
        ),
        "Logistic Regression": Pipeline(
            [
                ("tfidf", clone(vectorizer)),
                ("model", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)),
            ]
        ),
        "Random Forest": Pipeline(
            [
                ("tfidf", clone(vectorizer)),
                ("model", RandomForestClassifier(
                    n_estimators=300,
                    max_depth=None,
                    min_samples_split=2,
                    class_weight="balanced",
                    n_jobs=-1,
                    random_state=RANDOM_STATE,
                )),
            ]
        ),
    }


def evaluate_predictions(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    """Compute the main classification metrics."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, pos_label="positive", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, pos_label="positive", zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, pos_label="positive", zero_division=0)),
    }


def train_and_evaluate_models(
    X_train: pd.Series,
    X_test: pd.Series,
    y_train: pd.Series,
    y_test: pd.Series,
) -> list[ModelResult]:
    """Train the candidate models and evaluate them on the holdout set."""
    results: list[ModelResult] = []

    for model_name, pipeline in build_model_pipelines().items():
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        if hasattr(pipeline, "predict_proba"):
            proba_frame = pipeline.predict_proba(X_test)
        else:
            proba_frame = np.zeros((len(X_test), 2))

        results.append(
            ModelResult(
                name=model_name,
                pipeline=pipeline,
                metrics=evaluate_predictions(y_test, y_pred),
                confusion_matrix=confusion_matrix(y_test, y_pred, labels=["negative", "positive"]),
                y_pred=y_pred,
                y_proba=proba_frame,
            )
        )

    return sorted(results, key=lambda item: item.metrics["f1_score"], reverse=True)


def save_metrics(results: list[ModelResult]) -> pd.DataFrame:
    """Save the model metrics summary table."""
    metrics_df = pd.DataFrame(
        [{"model": result.name, **result.metrics} for result in results]
    ).sort_values("f1_score", ascending=False)
    metrics_df.to_csv(REPORTS_DIR / "metrics_summary.csv", index=False)
    return metrics_df


def save_confusion_matrices(results: list[ModelResult]) -> None:
    """Save a confusion matrix figure for each evaluated model."""
    for result in results:
        fig, axis = plt.subplots(figsize=(5, 4))
        ConfusionMatrixDisplay(
            confusion_matrix=result.confusion_matrix,
            display_labels=["negative", "positive"],
        ).plot(cmap="Blues", ax=axis, colorbar=False)
        axis.set_title(f"Confusion Matrix - {result.name}")
        plt.tight_layout()
        file_name = f"confusion_matrix_{result.name.lower().replace(' ', '_')}.png"
        plt.savefig(FIGURES_DIR / file_name, dpi=180)
        plt.close()


def save_sentiment_distribution(df: pd.DataFrame) -> None:
    """Save a bar chart of sentiment counts."""
    order = df["sentiment"].value_counts().index.tolist()
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x="sentiment", hue="sentiment", order=order, palette="Set2", legend=False)
    plt.title("Distribusi Sentimen Review Gojek")
    plt.xlabel("Sentimen")
    plt.ylabel("Jumlah Review")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "sentiment_distribution.png", dpi=180)
    plt.close()


def get_top_terms(df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    """Compute the most frequent cleaned tokens across the dataset."""
    all_tokens = " ".join(df["clean_text"].fillna("")).split()
    token_series = pd.Series(all_tokens, dtype="object")
    if token_series.empty:
        return pd.DataFrame(columns=["term", "count"])
    return (
        token_series.value_counts()
        .head(top_n)
        .rename_axis("term")
        .reset_index(name="count")
    )


def save_top_terms_chart(top_terms_df: pd.DataFrame) -> None:
    """Save a horizontal bar chart of the top tokens."""
    if top_terms_df.empty:
        return
    chart_df = top_terms_df.sort_values("count", ascending=True)
    plt.figure(figsize=(10, 7))
    plt.barh(chart_df["term"], chart_df["count"], color="#2f6b8a")
    plt.title("Top 20 Kata Paling Sering Muncul")
    plt.xlabel("Frekuensi")
    plt.ylabel("Kata")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "top_terms.png", dpi=180)
    plt.close()
    top_terms_df.to_csv(REPORTS_DIR / "top_terms.csv", index=False)


def save_wordcloud(df: pd.DataFrame, sentiment_label: str) -> None:
    """Generate a wordcloud for a specific sentiment class."""
    text = " ".join(df.loc[df["sentiment"] == sentiment_label, "clean_text"].fillna(""))
    if not text.strip():
        return

    wordcloud = WordCloud(
        width=1200,
        height=700,
        background_color="white",
        colormap="viridis" if sentiment_label == "positive" else "Reds",
    ).generate(text)

    plt.figure(figsize=(12, 7))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title(f"Wordcloud Review {sentiment_label.title()}")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"wordcloud_{sentiment_label}.png", dpi=180)
    plt.close()


def save_error_analysis(
    best_result: ModelResult,
    X_test: pd.Series,
    y_test: pd.Series,
    original_df: pd.DataFrame,
) -> pd.DataFrame:
    """Save misclassified review examples from the best model."""
    test_df = original_df.loc[X_test.index, ["review_text", "clean_text", "sentiment"]].copy()
    test_df["predicted_sentiment"] = best_result.y_pred

    if best_result.y_proba.size:
        class_index = list(best_result.pipeline.named_steps["model"].classes_).index("positive")
        test_df["positive_probability"] = best_result.y_proba[:, class_index]
    else:
        test_df["positive_probability"] = np.nan

    error_df = test_df.loc[test_df["sentiment"] != test_df["predicted_sentiment"]].copy()
    error_df["possible_reason"] = np.where(
        error_df["review_text"].str.contains(r"\b(?:tapi|namun|tp|padahal)\b", case=False, regex=True),
        "Review mengandung opini campuran atau kontras seperti 'tapi/namun'.",
        "Kemungkinan ada sarkasme, typo, slang, atau konteks domain yang sulit ditangkap model.",
    )
    error_df = error_df.sort_values("positive_probability", ascending=False)
    error_df.to_csv(REPORTS_DIR / "error_analysis_examples.csv", index=False)
    return error_df


def build_dataset_overview(df: pd.DataFrame) -> dict[str, object]:
    """Create a compact dataset summary for notebook and README use."""
    return {
        "row_count": int(len(df)),
        "column_count": int(df.shape[1]),
        "columns": df.columns.tolist(),
        "missing_values": df.isna().sum().to_dict(),
    }


def build_insight_summary(processed_df: pd.DataFrame, best_result: ModelResult, error_df: pd.DataFrame) -> str:
    """Generate a business-facing Markdown summary."""
    sentiment_counts = processed_df["sentiment"].value_counts()
    top_negative_terms = get_top_terms(processed_df.loc[processed_df["sentiment"] == "negative"], top_n=10)
    top_positive_terms = get_top_terms(processed_df.loc[processed_df["sentiment"] == "positive"], top_n=10)

    negative_terms_text = ", ".join(top_negative_terms["term"].tolist()[:8]) or "belum tersedia"
    positive_terms_text = ", ".join(top_positive_terms["term"].tolist()[:8]) or "belum tersedia"

    summary = f"""# Gojek Review Insight Summary

## Ringkasan Dataset

- Total review setelah preprocessing: `{len(processed_df):,}`
- Distribusi sentimen: `{sentiment_counts.to_dict()}`
- Model terbaik berdasarkan F1-score: `{best_result.name}`
- F1-score terbaik: `{best_result.metrics['f1_score']:.4f}`

## Keluhan Utama Pengguna

Kata yang paling sering muncul pada review negatif mengarah ke tema seperti: `{negative_terms_text}`.
Secara praktis, ini biasanya berkaitan dengan masalah performa aplikasi, proses login, promo, pembayaran, atau pengalaman pemesanan yang tidak konsisten.

## Aspek yang Disukai Pengguna

Review positif lebih sering menonjolkan kata seperti: `{positive_terms_text}`.
Ini biasanya menunjukkan apresiasi pengguna terhadap kemudahan penggunaan, kecepatan layanan, promo yang menarik, dan manfaat aplikasi dalam aktivitas harian.

## Error Analysis

- Jumlah review yang salah diprediksi model terbaik: `{len(error_df):,}`
- Pola umum kesalahan: review bercampur antara pujian dan keluhan, bahasa gaul, typo, dan konteks sarkastik.

## Rekomendasi Bisnis

1. Prioritaskan investigasi pada tema negatif yang paling dominan di `top_terms.csv` dan `wordcloud_negative.png`.
2. Audit perjalanan pengguna pada area yang paling sering dikeluhkan seperti pembayaran, promo, atau stabilitas aplikasi.
3. Pertahankan elemen yang sering dipuji pengguna dan jadikan itu bahan komunikasi produk.
4. Gunakan review salah prediksi sebagai sumber tambahan untuk memperkaya kamus normalisasi slang Bahasa Indonesia.
"""

    (REPORTS_DIR / "business_insights.md").write_text(summary, encoding="utf-8")
    return summary


def save_metadata(dataset_path: Path, metrics_df: pd.DataFrame, best_result: ModelResult) -> None:
    """Store compact metadata used by the Streamlit app and README."""
    metadata = {
        "dataset_file": str(dataset_path.relative_to(PROJECT_ROOT)),
        "best_model": best_result.name,
        "selection_metric": "f1_score",
        "metrics": metrics_df.to_dict(orient="records"),
        "generated_files": {
            "processed_data": str((PROCESSED_DATA_DIR / "gojek_reviews_processed.csv").relative_to(PROJECT_ROOT)),
            "metrics_summary": str((REPORTS_DIR / "metrics_summary.csv").relative_to(PROJECT_ROOT)),
            "error_analysis": str((REPORTS_DIR / "error_analysis_examples.csv").relative_to(PROJECT_ROOT)),
        },
    }
    (REPORTS_DIR / "project_summary.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def save_best_model(best_result: ModelResult, modeling_df: pd.DataFrame) -> None:
    """Retrain the best model on the full modeling dataset and save the artifact."""
    full_pipeline = clone(best_result.pipeline)
    full_pipeline.fit(modeling_df["clean_text"], modeling_df["sentiment"])
    joblib.dump(full_pipeline, MODELS_DIR / "best_sentiment_model.joblib")


def build_project_outputs(csv_path: str | Path | None = None) -> dict[str, object]:
    """Run the full project workflow from raw CSV to saved artifacts."""
    ensure_directories()
    dataset_path = find_dataset_path(csv_path)
    raw_df = load_dataset(dataset_path)
    processed_df = preprocess_reviews(raw_df)
    modeling_df = prepare_modeling_data(processed_df)
    X_train, X_test, y_train, y_test = split_dataset(modeling_df)

    results = train_and_evaluate_models(X_train, X_test, y_train, y_test)
    best_result = results[0]
    metrics_df = save_metrics(results)

    save_confusion_matrices(results)
    save_sentiment_distribution(processed_df)
    top_terms_df = get_top_terms(processed_df)
    save_top_terms_chart(top_terms_df)
    save_wordcloud(processed_df, "positive")
    save_wordcloud(processed_df, "negative")

    error_df = save_error_analysis(best_result, X_test, y_test, modeling_df)
    insight_summary = build_insight_summary(processed_df, best_result, error_df)
    save_best_model(best_result, modeling_df)
    save_metadata(dataset_path, metrics_df, best_result)

    overview = build_dataset_overview(processed_df)
    return {
        "dataset_path": dataset_path,
        "dataset_overview": overview,
        "processed_data": processed_df,
        "modeling_data": modeling_df,
        "metrics": metrics_df,
        "best_model": best_result.name,
        "best_result": best_result,
        "error_analysis": error_df,
        "insight_summary": insight_summary,
    }


def print_run_summary(result: dict[str, object]) -> None:
    """Print a compact CLI summary after the pipeline finishes."""
    metrics_df: pd.DataFrame = result["metrics"]  # type: ignore[assignment]
    best_model = result["best_model"]
    dataset_path: Path = result["dataset_path"]  # type: ignore[assignment]

    print("Gojek Review Insight pipeline completed successfully.")
    print(f"Dataset file: {dataset_path}")
    print(f"Best model: {best_model}")
    print(metrics_df.to_string(index=False))


def build_readme_highlight(result: dict[str, object]) -> str:
    """Create a short highlight block that can be pasted into reports or docs."""
    metrics_df: pd.DataFrame = result["metrics"]  # type: ignore[assignment]
    best_row = metrics_df.iloc[0]
    return textwrap.dedent(
        f"""
        Best model: {best_row['model']}
        Accuracy : {best_row['accuracy']:.4f}
        Precision: {best_row['precision']:.4f}
        Recall   : {best_row['recall']:.4f}
        F1-score : {best_row['f1_score']:.4f}
        """
    ).strip()


def main(csv_path: str | Path | None = None) -> dict[str, object]:
    """Run the end-to-end pipeline and print the result summary."""
    result = build_project_outputs(csv_path=csv_path)
    print_run_summary(result)
    return result


if __name__ == "__main__":
    main()
