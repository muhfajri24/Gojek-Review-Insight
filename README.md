# Gojek Review Insight

Gojek Review Insight is a portfolio-ready data science project that analyzes Indonesian-language Gojek app reviews to understand customer sentiment, surface recurring complaint themes, and translate the findings into presentation-friendly business insights.

The repository is designed for a clean and professional portfolio presentation: modular pipeline, notebook walkthrough, multi-model comparison, visual reporting, error analysis, and a lightweight Streamlit demo app.

## Problem Statement

User reviews on Google Play and the App Store contain valuable signals about customer experience. The challenge is that large review volumes make manual analysis inefficient. This project focuses on three main questions:

1. What is the sentiment distribution of Gojek app reviews?
2. Which complaint themes appear most often in negative reviews?
3. Which machine learning model performs best for classifying Indonesian-language review sentiment?

## Dataset

This project is based on the following Kaggle references:

- Notebook: `najwaputrif/sentiment-analysis-review-aplikasi-gojek`
- Dataset: `ucupsedaya/gojek-app-reviews-bahasa-indonesia`
- Main file: `GojekAppReviewV4.0.0-V4.9.3_Cleaned.csv`

Project-specific adjustments used here:

- The pipeline automatically detects review, rating, and sentiment columns when they use common names such as `content`, `review`, `ulasan`, `rating`, or `sentiment`.
- If a sentiment column is not available, labels are inferred from ratings:
  - `4-5` = `positive`
  - `1-2` = `negative`
  - `3` = `neutral`
- To stay aligned with the reference notebook, the pipeline focuses on reviews whose `appVersion` starts with `4.8` when that column is available.
- For supervised modeling, the `neutral` class is excluded from training so the classifier stays focused on `positive` vs `negative`.

## Tools

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Sastrawi
- WordCloud
- Streamlit
- Joblib

## Workflow

1. Load the Gojek review dataset from CSV.
2. Perform a dataset overview: row count, column structure, missing values, sentiment distribution, and example reviews.
3. Clean Indonesian text with lowercasing, symbol and number removal, emoji cleanup, stopword removal, and Sastrawi stemming.
4. Extract text features with TF-IDF.
5. Train and compare three models:
   - Naive Bayes
   - Logistic Regression
   - Random Forest
6. Evaluate model performance using:
   - accuracy
   - precision
   - recall
   - f1-score
   - confusion matrix
7. Select the best model based on `F1-score`.
8. Review error analysis for misclassified examples.
9. Summarize business insights and application improvement recommendations.
10. Present the result through a simple Streamlit demo.

## Final Result

Latest processed-run results from the Kaggle dataset:

- Total reviews after preprocessing: `6,152`
- Sentiment distribution: `3,131 positive`, `2,670 negative`, `351 neutral`
- Best model: `Naive Bayes`
- Selection metric: `F1-score`

### Model Comparison

| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Naive Bayes | 0.8794 | 0.9372 | 0.8325 | 0.8818 |
| Logistic Regression | 0.8725 | 0.9180 | 0.8389 | 0.8767 |
| Random Forest | 0.8562 | 0.8966 | 0.8293 | 0.8616 |

The best model is selected by `F1-score`, not by accuracy alone.

## Main Insights

- The most common complaint terms include `driver`, `gopay`, `bayar`, `lama`, and `malah`.
- This suggests that negative reviews cluster around payment friction, application stability, and inconsistent service experiences.
- Positive reviews more often emphasize terms such as `bantu`, `bagus`, `baik`, `mudah`, `mantap`, and `cepat`.
- This indicates that users value ease of use, practical utility, and service quality when the product experience runs smoothly.
- Error analysis shows many difficult cases involve mixed sentiment, slang, typos, or contrastive phrasing such as "good, but...".

## Project Structure

```text
Gojek-Review-Insight/
|-- app/
|-- data/
|   |-- processed/
|   |-- raw/
|-- notebook/
|   |-- gojek_review_insight_walkthrough.ipynb
|   |-- gojek_review_insight_walkthrough.py
|-- output/
|   |-- figures/
|   |-- models/
|   |-- reports/
|-- src/
|   |-- __init__.py
|   |-- sentiment_pipeline.py
|-- README.md
|-- requirements.txt
|-- streamlit_app.py
```

## Output Files

After the pipeline runs, the project exports:

- `data/processed/gojek_reviews_processed.csv`
- `output/reports/metrics_summary.csv`
- `output/reports/top_terms.csv`
- `output/reports/error_analysis_examples.csv`
- `output/reports/business_insights.md`
- `output/reports/project_summary.json`
- `output/models/best_sentiment_model.joblib`
- `output/figures/sentiment_distribution.png`
- `output/figures/top_terms.png`
- `output/figures/wordcloud_positive.png`
- `output/figures/wordcloud_negative.png`
- `output/figures/confusion_matrix_naive_bayes.png`
- `output/figures/confusion_matrix_logistic_regression.png`
- `output/figures/confusion_matrix_random_forest.png`

## How to Run

### 1. Install dependencies

```powershell
pip install -r requirements.txt
```

### 2. Make sure the dataset is available

Place the CSV file in one of the following locations:

- `data/raw/gojek_reviews.csv`
- or keep the original Kaggle filename inside `data/raw/`

### 3. Run the pipeline

```powershell
python -m src.sentiment_pipeline
```

### 4. Launch the Streamlit app

```powershell
streamlit run streamlit_app.py
```

## Notebook and Colab

The main walkthrough notebook is available in two formats:

- `notebook/gojek_review_insight_walkthrough.py`
- `notebook/gojek_review_insight_walkthrough.ipynb`

The notebook has been adjusted to work more reliably in:

- VS Code notebooks
- local Jupyter
- Google Colab

If it is executed in Colab without the full project structure, the notebook will attempt to clone this repository:

- `muhfajri24/Gojek-Review-Insight`

## Visuals

Main visual outputs are available in:

- `output/figures/sentiment_distribution.png`
- `output/figures/top_terms.png`
- `output/figures/wordcloud_positive.png`
- `output/figures/wordcloud_negative.png`

## Why This Project Works Well for a Portfolio

- The topic is relevant and easy for recruiters to understand.
- It focuses on Indonesian-language NLP, which gives it a distinct local use case.
- It covers EDA, preprocessing, TF-IDF, model comparison, evaluation, and business insight.
- It includes error analysis instead of optimizing only for accuracy.
- It provides a Streamlit app for interactive demo use.
- The latest outputs are already included, so the repository feels complete.

## Future Improvements

- Expand Indonesian slang normalization coverage.
- Test additional models such as Linear SVM or IndoBERT.
- Add topic modeling to explore complaint themes more deeply.
- Connect the output to a BI dashboard.
- Deploy the Streamlit app online.

## References

- Kaggle notebook: `najwaputrif/sentiment-analysis-review-aplikasi-gojek`
- Kaggle dataset: `ucupsedaya/gojek-app-reviews-bahasa-indonesia`
- GitHub repository: `muhfajri24/Gojek-Review-Insight`
