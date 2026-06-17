# %%
"""
Notebook-style walkthrough for the Gojek Review Insight project.

Open this file in VS Code and run each `# %%` cell step by step to present the
analysis like a Jupyter notebook while keeping the project logic modular.
"""

from pathlib import Path
import subprocess
import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from IPython.display import Markdown, display


def resolve_project_root() -> Path:
    """Resolve the project root for local notebooks, Jupyter, or Colab."""
    candidates: list[Path] = []

    if "__file__" in globals():
        candidates.append(Path(__file__).resolve().parent)

    cwd = Path.cwd().resolve()
    candidates.extend([cwd, *cwd.parents])

    for candidate in candidates:
        if (candidate / "src" / "sentiment_pipeline.py").exists():
            return candidate

    colab_target = Path("/content/Gojek-Review-Insight")
    if "google.colab" in sys.modules and not colab_target.exists():
        subprocess.run(
            ["git", "clone", "https://github.com/muhfajri24/Gojek-Review-Insight.git", str(colab_target)],
            check=True,
        )

    if (colab_target / "src" / "sentiment_pipeline.py").exists():
        return colab_target

    raise FileNotFoundError(
        "Project root tidak ditemukan. Jalankan notebook dari folder project, "
        "atau pastikan repo `muhfajri24/Gojek-Review-Insight` tersedia di runtime."
    )


def ensure_dependencies(project_root: Path) -> None:
    """Install project dependencies automatically in Google Colab."""
    if "google.colab" not in sys.modules:
        return

    requirements_path = project_root / "requirements.txt"
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", str(requirements_path)],
        check=True,
    )


PROJECT_ROOT = resolve_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
ensure_dependencies(PROJECT_ROOT)

from src.sentiment_pipeline import (  # noqa: E402
    FIGURES_DIR,
    REPORTS_DIR,
    build_project_outputs,
    build_readme_highlight,
    find_dataset_path,
    get_top_terms,
    load_dataset,
    preprocess_reviews,
    prepare_modeling_data,
    save_wordcloud,
    standardize_review_dataset,
)


sns.set_theme(style="whitegrid")
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 160)


def show_section(title: str, subtitle: str = "") -> None:
    text = f"## {title}"
    if subtitle:
        text += f"\n\n{subtitle}"
    display(Markdown(text))


# %%
show_section(
    "1. Dataset Loading",
    "Pastikan file CSV Kaggle sudah diletakkan di `data/raw/gojek_reviews.csv` sebelum menjalankan walkthrough ini.",
)
dataset_path = find_dataset_path()
raw_df = load_dataset(dataset_path)

print("Dataset path:", dataset_path)
print("Jumlah data mentah:", len(raw_df))
display(raw_df.head())


# %%
show_section("2. Struktur Kolom", "Lihat nama kolom dan beberapa informasi dasar dataset.")
standardized_df = standardize_review_dataset(raw_df)
overview_df = pd.DataFrame(
    {
        "column": standardized_df.columns,
        "dtype": standardized_df.dtypes.astype(str).values,
        "missing_count": standardized_df.isna().sum().values,
        "unique_count": standardized_df.nunique(dropna=False).values,
    }
)
display(overview_df)


# %%
show_section("3. Missing Values", "Identifikasi kolom yang perlu perhatian saat data cleaning.")
missing_df = (
    standardized_df.isna()
    .sum()
    .rename("missing_count")
    .reset_index()
    .rename(columns={"index": "column"})
    .sort_values("missing_count", ascending=False)
)
display(missing_df)

plt.figure(figsize=(10, 5))
sns.barplot(data=missing_df, x="column", y="missing_count", hue="column", palette="crest", legend=False)
plt.title("Missing Values per Column")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.show()


# %%
show_section("4. Preprocessing", "Lowercase, hapus simbol, stopword Bahasa Indonesia, dan stemming Sastrawi.")
processed_df = preprocess_reviews(raw_df)

print("Jumlah data setelah preprocessing:", len(processed_df))
display(processed_df[["review_text", "clean_text", "sentiment"]].head(10))


# %%
show_section("5. Distribusi Sentimen", "Periksa distribusi rating atau sentimen yang berhasil diinfer.")
sentiment_counts = processed_df["sentiment"].value_counts().rename_axis("sentiment").reset_index(name="count")
display(sentiment_counts)

plt.figure(figsize=(8, 5))
sns.countplot(data=processed_df, x="sentiment", hue="sentiment", palette="Set2", legend=False)
plt.title("Distribusi Sentimen Review Gojek")
plt.tight_layout()
plt.show()


# %%
show_section("6. Contoh Review Positif dan Negatif", "Gunakan contoh ini untuk cerita bisnis saat presentasi.")
positive_examples = processed_df.loc[processed_df["sentiment"] == "positive", ["review_text"]].head(5)
negative_examples = processed_df.loc[processed_df["sentiment"] == "negative", ["review_text"]].head(5)

display(Markdown("### Contoh Review Positif"))
display(positive_examples)
display(Markdown("### Contoh Review Negatif"))
display(negative_examples)


# %%
show_section("7. Kata Paling Sering Muncul", "Lihat top 20 token setelah preprocessing.")
top_terms_df = get_top_terms(processed_df, top_n=20)
display(top_terms_df)

plt.figure(figsize=(10, 7))
chart_df = top_terms_df.sort_values("count", ascending=True)
plt.barh(chart_df["term"], chart_df["count"], color="#2f6b8a")
plt.title("Top 20 Kata Paling Sering Muncul")
plt.tight_layout()
plt.show()


# %%
show_section("8. Wordcloud", "Buat wordcloud untuk review positif dan negatif.")
save_wordcloud(processed_df, "positive")
save_wordcloud(processed_df, "negative")

positive_image = plt.imread(FIGURES_DIR / "wordcloud_positive.png")
negative_image = plt.imread(FIGURES_DIR / "wordcloud_negative.png")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
axes[0].imshow(positive_image)
axes[0].axis("off")
axes[0].set_title("Wordcloud Positif")
axes[1].imshow(negative_image)
axes[1].axis("off")
axes[1].set_title("Wordcloud Negatif")
plt.tight_layout()
plt.show()


# %%
show_section("9. Data untuk Modeling", "Fokuskan model pada kelas positive vs negative.")
modeling_df = prepare_modeling_data(processed_df)
display(modeling_df["sentiment"].value_counts().to_frame(name="count"))
display(modeling_df[["clean_text", "sentiment"]].head())


# %%
show_section("10. Training dan Evaluasi Model", "Bandingkan Naive Bayes, Logistic Regression, dan Random Forest.")
result = build_project_outputs()
metrics_df = result["metrics"]
display(metrics_df)


# %%
show_section("11. Visual Review", "Tampilkan confusion matrix dan insight files yang sudah diekspor.")
for image_path in sorted(FIGURES_DIR.glob("confusion_matrix_*.png")):
    print(image_path.name)

error_analysis_df = pd.read_csv(REPORTS_DIR / "error_analysis_examples.csv")
display(error_analysis_df.head(10))


# %%
show_section("12. Business Insight", "Ringkasan ini bisa langsung dipakai untuk README atau presentasi.")
print(build_readme_highlight(result))
print()
print((REPORTS_DIR / "business_insights.md").read_text(encoding="utf-8"))
