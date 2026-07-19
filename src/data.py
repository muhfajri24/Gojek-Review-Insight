from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable
import pandas as pd
from src.preprocessing import basic_clean_text

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
DEFAULT_CSV_NAME = "gojek_reviews.csv"
TEXT_COLUMN_CANDIDATES = ("content", "review", "review_text", "text", "ulasan", "komentar", "comment")
RATING_COLUMN_CANDIDATES = ("rating", "score", "star", "stars", "nilai")
SENTIMENT_COLUMN_CANDIDATES = ("sentiment", "label", "polarity", "class", "kelas")


def find_dataset_path(csv_path: str | Path | None = None) -> Path:
    if csv_path is not None:
        path = Path(csv_path)
        path = path if path.is_absolute() else PROJECT_ROOT / path
        if not path.is_file():
            raise FileNotFoundError(f"Dataset file not found: {path}")
        return path
    preferred = RAW_DATA_DIR / DEFAULT_CSV_NAME
    if preferred.is_file():
        return preferred
    candidates = sorted(RAW_DATA_DIR.glob("*.csv"))
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        names = ", ".join(path.name for path in candidates)
        raise FileNotFoundError(f"Multiple CSV files found in data/raw ({names}). Rename the intended file to {DEFAULT_CSV_NAME} or pass csv_path explicitly.")
    raise FileNotFoundError(f"No dataset found. Place the review CSV at data/raw/{DEFAULT_CSV_NAME} or pass a project-relative csv_path.")


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result.columns = [re.sub(r"[^a-z0-9]+", "_", str(column).strip().lower()).strip("_") for column in result.columns]
    return result


def _first_column(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    lookup = {column.lower(): column for column in columns}
    return next((lookup[name] for name in candidates if name in lookup), None)


def validate_and_standardize_dataset(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        raise ValueError("Dataset is empty.")
    result = normalize_column_names(df)
    text_column = _first_column(result.columns, TEXT_COLUMN_CANDIDATES)
    rating_column = _first_column(result.columns, RATING_COLUMN_CANDIDATES)
    sentiment_column = _first_column(result.columns, SENTIMENT_COLUMN_CANDIDATES)
    if text_column is None:
        raise ValueError("Missing review-text column. Expected one of: " + ", ".join(TEXT_COLUMN_CANDIDATES))
    if rating_column is None and sentiment_column is None:
        raise ValueError("Dataset must contain a rating or sentiment column.")
    rename_map = {text_column: "review_text"}
    if rating_column:
        rename_map[rating_column] = "rating"
    if sentiment_column:
        rename_map[sentiment_column] = "sentiment"
    result = result.rename(columns=rename_map)
    result["review_text"] = result["review_text"].fillna("").astype(str).str.strip()
    result = result.loc[result["review_text"].ne("")].copy()
    if "rating" in result:
        result["rating"] = pd.to_numeric(result["rating"], errors="coerce")
    return result


def load_dataset(csv_path: str | Path | None = None) -> tuple[pd.DataFrame, Path]:
    path = find_dataset_path(csv_path)
    try:
        raw = pd.read_csv(path)
    except Exception as exc:
        raise ValueError(f"Could not read dataset CSV: {path}") from exc
    return validate_and_standardize_dataset(raw), path


def rating_to_sentiment(rating: object) -> str:
    """Map ratings 1-2 to negative, 3 to neutral, and 4-5 to positive."""
    if pd.isna(rating) or float(rating) not in {1.0, 2.0, 3.0, 4.0, 5.0}:
        raise ValueError(f"Unsupported rating value: {rating!r}. Expected an integer from 1 to 5.")
    value = int(float(rating))
    return "negative" if value <= 2 else "neutral" if value == 3 else "positive"


def normalize_sentiment_label(value: object) -> str | None:
    if pd.isna(value):
        return None
    mapping = {"positif": "positive", "positive": "positive", "negatif": "negative", "negative": "negative", "netral": "neutral", "neutral": "neutral"}
    return mapping.get(str(value).strip().lower())


def assign_sentiment_labels(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    if "rating" in result:
        invalid = result["rating"].isna() | ~result["rating"].isin([1, 2, 3, 4, 5])
        if invalid.any():
            values = result.loc[invalid, "rating"].drop_duplicates().head(10).tolist()
            raise ValueError(f"Unsupported or missing ratings found: {values}")
        result["sentiment"] = result["rating"].map(rating_to_sentiment)
        result["label_source"] = "rating_rule"
    else:
        result["sentiment"] = result["sentiment"].map(normalize_sentiment_label)
        if result["sentiment"].isna().any():
            raise ValueError("Unsupported sentiment labels found.")
        result["label_source"] = "dataset_label"
    return result


def prepare_dataset(df: pd.DataFrame, version_prefix: str | None = "4.8") -> tuple[pd.DataFrame, dict[str, int]]:
    if version_prefix and "appversion" in df:
        mask = df["appversion"].astype(str).str.startswith(version_prefix, na=False)
        if not mask.any():
            raise ValueError(f"No reviews match app-version prefix {version_prefix!r}.")
        df = df.loc[mask].copy()
    else:
        df = df.copy()
    rows_after_filter = len(df)
    result = assign_sentiment_labels(df)
    exact_duplicates = int(result.duplicated(subset=["review_text"]).sum())
    result = result.drop_duplicates(subset=["review_text"], keep="first").copy()
    result["split_group"] = result["review_text"].map(basic_clean_text)
    empty_clean = int(result["split_group"].eq("").sum())
    result = result.loc[result["split_group"].ne("")].copy()
    group_labels = result.groupby("split_group")["sentiment"].nunique()
    conflicting_groups = set(group_labels[group_labels > 1].index)
    conflicting_rows = int(result["split_group"].isin(conflicting_groups).sum())
    result = result.loc[~result["split_group"].isin(conflicting_groups)].copy()
    result["review_length"] = result["review_text"].str.len()
    result = result.reset_index(drop=True)
    stats = {"rows_after_version_filter": rows_after_filter, "exact_duplicate_reviews_removed": exact_duplicates, "empty_after_basic_cleaning_removed": empty_clean, "conflicting_normalized_text_rows_removed": conflicting_rows, "usable_rows": len(result)}
    return result, stats
