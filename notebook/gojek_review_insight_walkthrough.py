# %%
"""Optional notebook-style walkthrough for the primary Python pipeline."""

from pathlib import Path
import sys
import pandas as pd


def resolve_project_root() -> Path:
    candidates = [Path.cwd().resolve(), *Path.cwd().resolve().parents]
    if "__file__" in globals():
        candidates.insert(0, Path(__file__).resolve().parent.parent)
    for candidate in candidates:
        if (candidate / "src" / "sentiment_pipeline.py").is_file():
            return candidate
    raise FileNotFoundError("Run this walkthrough from the project repository.")


PROJECT_ROOT = resolve_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import find_dataset_path, load_dataset  # noqa: E402
from src.sentiment_pipeline import run_pipeline  # noqa: E402

# %%
dataset_path = find_dataset_path()
raw_df, _ = load_dataset(dataset_path)
print("Dataset:", dataset_path.relative_to(PROJECT_ROOT))
print("Validated raw shape:", raw_df.shape)
raw_df.head()

# %%
result = run_pipeline()
result["comparison"]

# %%
pd.read_csv(PROJECT_ROOT / "output" / "metrics" / "classification_report_best_model.csv")

# %%
pd.read_csv(PROJECT_ROOT / "output" / "insights" / "complaint_themes.csv")

# %%
print((PROJECT_ROOT / "output" / "insights" / "business_summary.md").read_text(encoding="utf-8"))
