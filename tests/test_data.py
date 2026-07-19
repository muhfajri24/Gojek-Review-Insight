import pandas as pd
import pytest
from src.data import rating_to_sentiment, validate_and_standardize_dataset


def test_required_columns_are_validated():
    with pytest.raises(ValueError, match="review-text"):
        validate_and_standardize_dataset(pd.DataFrame({"score": [5]}))


def test_rating_mapping():
    assert [rating_to_sentiment(value) for value in range(1, 6)] == ["negative", "negative", "neutral", "positive", "positive"]


def test_invalid_rating_is_rejected():
    with pytest.raises(ValueError, match="Unsupported rating"):
        rating_to_sentiment(6)
