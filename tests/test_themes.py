import pandas as pd
from src.themes import categorize_complaint_themes


def test_theme_does_not_duplicate_review_within_theme():
    frame = pd.DataFrame({"review_text": ["gopay bayar gagal bayar", "bagus"], "sentiment": ["negative", "positive"]})
    themes, _ = categorize_complaint_themes(frame)
    payment = themes.loc[themes["theme"] == "payment problems"].iloc[0]
    assert payment["review_count"] == 1
