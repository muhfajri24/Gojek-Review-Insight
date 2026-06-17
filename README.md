# Gojek Review Insight

Gojek Review Insight adalah project portofolio Data Scientist yang menganalisis ulasan pengguna aplikasi Gojek berbahasa Indonesia untuk memahami sentimen pengguna, menemukan pola keluhan yang paling sering muncul, dan menerjemahkan hasil analisis menjadi insight bisnis yang mudah dipresentasikan.

Project ini dibuat agar nyaman untuk portfolio GitHub junior Data Scientist: struktur rapi, pipeline modular, notebook walkthrough, perbandingan beberapa model, visualisasi, error analysis, dan aplikasi Streamlit sederhana untuk demo.

## Problem Statement

Ulasan pengguna di Google Play atau App Store menyimpan sinyal penting tentang pengalaman pelanggan. Tantangannya, jumlah review yang besar membuat analisis manual tidak efisien. Project ini menjawab tiga pertanyaan utama:

1. Bagaimana distribusi sentimen pengguna terhadap aplikasi Gojek?
2. Keluhan apa yang paling sering muncul pada review negatif?
3. Model machine learning mana yang paling baik untuk mengklasifikasikan sentimen review berbahasa Indonesia?

## Dataset

Project ini menggunakan referensi dari Kaggle notebook:

- Notebook: `najwaputrif/sentiment-analysis-review-aplikasi-gojek`
- Dataset: `ucupsedaya/gojek-app-reviews-bahasa-indonesia`
- File utama: `GojekAppReviewV4.0.0-V4.9.3_Cleaned.csv`

Penyesuaian yang dipakai di project ini:

- Pipeline otomatis mendeteksi kolom review, rating, dan sentiment jika namanya umum seperti `content`, `review`, `ulasan`, `rating`, atau `sentiment`.
- Jika kolom sentimen belum tersedia, label diinfer dari rating:
  - `4-5` = `positive`
  - `1-2` = `negative`
  - `3` = `neutral`
- Agar konsisten dengan notebook referensi, pipeline memfokuskan analisis pada review dengan `appVersion` awalan `4.8` saat kolom itu tersedia.
- Untuk supervised modeling, kelas `neutral` tidak dipakai pada training agar fokus klasifikasi tetap jelas pada `positive` vs `negative`.

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

1. Load dataset review Gojek dari file CSV.
2. Lakukan data overview: jumlah data, kolom, missing value, distribusi sentimen, dan contoh review.
3. Bersihkan teks Bahasa Indonesia dengan lowercase, hapus simbol, angka, emoji, stopword, dan stemming menggunakan Sastrawi.
4. Ekstraksi fitur teks menggunakan TF-IDF.
5. Latih dan bandingkan tiga model:
   - Naive Bayes
   - Logistic Regression
   - Random Forest
6. Evaluasi model menggunakan:
   - accuracy
   - precision
   - recall
   - f1-score
   - confusion matrix
7. Pilih model terbaik berdasarkan `F1-score`.
8. Tampilkan error analysis untuk review yang salah diprediksi.
9. Rangkum insight bisnis dan rekomendasi perbaikan aplikasi.
10. Sajikan demo interaktif lewat Streamlit.

## Final Result

Hasil run terbaru pada dataset Kaggle yang sudah diproses:

- Total review setelah preprocessing: `6,152`
- Distribusi sentimen: `3,131 positive`, `2,670 negative`, `351 neutral`
- Model terbaik: `Naive Bayes`
- Selection metric: `F1-score`

### Model Comparison

| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Naive Bayes | 0.8794 | 0.9372 | 0.8325 | 0.8818 |
| Logistic Regression | 0.8725 | 0.9180 | 0.8389 | 0.8767 |
| Random Forest | 0.8562 | 0.8966 | 0.8293 | 0.8616 |

Model terbaik dipilih berdasarkan `F1-score`, bukan hanya accuracy.

## Main Insights

- Keluhan utama pengguna paling sering berkaitan dengan kata seperti `driver`, `gopay`, `bayar`, `lama`, dan `malah`.
- Ini mengindikasikan tema utama pada review negatif berada di sekitar pembayaran, stabilitas aplikasi, dan pengalaman layanan yang tidak konsisten.
- Review positif lebih sering menonjolkan kata seperti `bantu`, `bagus`, `baik`, `mudah`, `mantap`, dan `cepat`.
- Ini menunjukkan pengguna menghargai kemudahan penggunaan, manfaat aplikasi, dan kualitas layanan saat pengalaman berjalan lancar.
- Error analysis menunjukkan cukup banyak review yang bersifat campuran, penuh slang, typo, atau konteks kontras seperti “bagus tapi…”.

## Project Structure

```text
Gojek-Review-Insight/
├── app/
├── data/
│   ├── processed/
│   └── raw/
├── notebook/
│   ├── gojek_review_insight_walkthrough.ipynb
│   └── gojek_review_insight_walkthrough.py
├── output/
│   ├── figures/
│   ├── models/
│   └── reports/
├── src/
│   ├── __init__.py
│   └── sentiment_pipeline.py
├── README.md
├── requirements.txt
└── streamlit_app.py
```

## Output Files

Setelah pipeline dijalankan, project mengekspor:

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

### 2. Pastikan dataset tersedia

Simpan file CSV di salah satu lokasi berikut:

- `data/raw/gojek_reviews.csv`
- atau biarkan dengan nama asli Kaggle di dalam folder `data/raw/`

### 3. Jalankan pipeline

```powershell
python -m src.sentiment_pipeline
```

### 4. Jalankan Streamlit app

```powershell
streamlit run streamlit_app.py
```

## Notebook and Colab

Notebook utama tersedia dalam dua format:

- `notebook/gojek_review_insight_walkthrough.py`
- `notebook/gojek_review_insight_walkthrough.ipynb`

Notebook sudah diperbaiki agar lebih aman untuk:

- VS Code notebook
- Jupyter lokal
- Google Colab

Jika dijalankan di Colab tanpa struktur project lengkap, notebook akan mencoba clone repo ini:

- `muhfajri24/Gojek-Review-Insight`

## Visuals

Visual utama tersedia di folder:

- `output/figures/sentiment_distribution.png`
- `output/figures/top_terms.png`
- `output/figures/wordcloud_positive.png`
- `output/figures/wordcloud_negative.png`

## Why This Project Is Strong for Portfolio

- Topiknya relevan dan mudah dipahami recruiter
- Fokus pada NLP Bahasa Indonesia
- Mencakup EDA, preprocessing, TF-IDF, model comparison, evaluasi, dan business insight
- Ada error analysis, bukan hanya mengejar accuracy
- Ada Streamlit app untuk demo interaktif
- Hasil run sudah tersedia sehingga repo tidak terasa kosong

## Future Improvements

- Tambahkan normalisasi slang Bahasa Indonesia yang lebih lengkap
- Coba model lain seperti Linear SVM atau IndoBERT
- Tambahkan topic modeling untuk eksplorasi tema keluhan
- Hubungkan output ke dashboard BI
- Deploy Streamlit app secara online

## References

- Kaggle notebook: `najwaputrif/sentiment-analysis-review-aplikasi-gojek`
- Kaggle dataset: `ucupsedaya/gojek-app-reviews-bahasa-indonesia`
- GitHub repository: `muhfajri24/Gojek-Review-Insight`
