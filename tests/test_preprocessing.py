import pytest
from src.preprocessing import NEGATIONS, basic_clean_text, linguistic_clean_text


def test_text_cleaning_returns_normalized_string():
    assert basic_clean_text("  BAGUS!!! https://example.com  ") == "bagus"


def test_empty_input_is_handled():
    assert basic_clean_text("") == ""


def test_repeated_characters_are_normalized():
    assert basic_clean_text("baguuuus") == "baguus"


@pytest.mark.parametrize("negation", sorted(NEGATIONS))
def test_negation_terms_are_preserved(negation):
    assert negation in linguistic_clean_text(f"{negation} bagus").split()
