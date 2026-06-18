# Gojek Review Insight

This project analyzes Indonesian-language Gojek app reviews to classify sentiment, surface recurring complaint themes, and translate review data into product insight.

## What This Project Does

- Loads and cleans Gojek review data from Kaggle
- Applies Indonesian text preprocessing and TF-IDF feature extraction
- Compares `Naive Bayes`, `Logistic Regression`, and `Random Forest`
- Highlights complaint patterns, model performance, and business recommendations
- Provides a lightweight Streamlit app for interactive review and testing

## Why It Matters

This project shows how Python can transform unstructured review text into measurable sentiment signals that help explain customer pain points, product quality, and service improvement priorities.

## Primary Workflow

Python is the main way to run this project.

```bash
pip install -r requirements.txt
python -m src.sentiment_pipeline
```

Optional demo app:

```bash
streamlit run streamlit_app.py
```

Main entrypoints:

- `src/sentiment_pipeline.py` for the sentiment analysis workflow
- `streamlit_app.py` for the demo interface

## Optional Notebook

Notebook files are included only as a secondary option for walkthroughs, recruiter demos, or quick testing.

- `notebook/gojek_review_insight_walkthrough.py`
- `notebook/gojek_review_insight_walkthrough.ipynb`
- `sentiment-analysis-review-aplikasi-gojek.ipynb`

## Dataset

Primary references:

- Kaggle dataset: `ucupsedaya/gojek-app-reviews-bahasa-indonesia`
- Kaggle notebook reference: `najwaputrif/sentiment-analysis-review-aplikasi-gojek`

Expected staged file: `data/raw/gojek_reviews.csv`

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

## Project Structure

```text
Gojek-Review-Insight/
|-- app/
|-- data/
|-- notebook/
|-- output/
|-- src/
|   `-- sentiment_pipeline.py
|-- streamlit_app.py
|-- requirements.txt
`-- README.md
```
