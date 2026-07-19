from __future__ import annotations

from pathlib import Path

from src.data import PROJECT_ROOT, load_dataset, prepare_dataset
from src.reporting import (
    export_error_analysis,
    export_feature_interpretation,
    generate_figures,
    save_model_and_metadata,
    save_model_evaluation,
    save_split_summary,
    write_business_summary,
)
from src.themes import categorize_complaint_themes
from src.training import ExperimentResult, create_shared_split, run_experiments

PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "output"
FIGURES_DIR = OUTPUT_DIR / "figures"
METRICS_DIR = OUTPUT_DIR / "metrics"
INSIGHTS_DIR = OUTPUT_DIR / "insights"
MODELS_DIR = PROJECT_ROOT / "models"


def ensure_directories() -> None:
    for directory in [
        PROCESSED_DATA_DIR, FIGURES_DIR, METRICS_DIR, INSIGHTS_DIR, MODELS_DIR
    ]:
        directory.mkdir(parents=True, exist_ok=True)


def run_pipeline(
    csv_path: str | Path | None = None,
    version_prefix: str | None = "4.8",
) -> dict[str, object]:
    ensure_directories()
    source_df, dataset_path = load_dataset(csv_path)
    dataset, preparation_stats = prepare_dataset(source_df, version_prefix)
    train_index, test_index = create_shared_split(dataset)
    train_df = dataset.iloc[train_index].copy()
    test_df = dataset.iloc[test_index].copy()

    split_summary = save_split_summary(
        dataset, train_df, test_df, preparation_stats, METRICS_DIR
    )
    dataset.drop(columns=["split_group"]).to_csv(
        PROCESSED_DATA_DIR / "gojek_reviews_processed.csv", index=False
    )

    results = run_experiments(train_df, test_df)
    best = results[0]
    comparison, classification = save_model_evaluation(
        results, best, test_df, METRICS_DIR
    )
    generate_figures(dataset, comparison, best, test_df, FIGURES_DIR)
    terms = export_feature_interpretation(best, INSIGHTS_DIR)
    themes, unmatched = categorize_complaint_themes(dataset, INSIGHTS_DIR)
    errors, error_summary = export_error_analysis(best, test_df, INSIGHTS_DIR)
    business_summary = write_business_summary(
        dataset, comparison, best, classification, themes, INSIGHTS_DIR
    )
    metadata = save_model_and_metadata(
        best, dataset_path, PROJECT_ROOT, MODELS_DIR, split_summary
    )
    return {
        "dataset": dataset,
        "train": train_df,
        "test": test_df,
        "comparison": comparison,
        "classification_report": classification,
        "best": best,
        "themes": themes,
        "unmatched": unmatched,
        "errors": errors,
        "error_summary": error_summary,
        "business_summary": business_summary,
        "terms": terms,
        "metadata": metadata,
    }


def main() -> None:
    result = run_pipeline()
    best: ExperimentResult = result["best"]  # type: ignore[assignment]
    print("Gojek Review Insight pipeline completed.")
    print(f"Best model: {best.model_name} ({best.preprocessing_variant})")
    print(f"Macro F1: {best.metrics['macro_f1']:.4f}")
    print(result["comparison"].to_string(index=False))


if __name__ == "__main__":
    main()
