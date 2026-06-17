# Gojek Review Insight

Gojek Review Insight adalah project portofolio Data Scientist yang menganalisis ulasan pengguna aplikasi Gojek berbahasa Indonesia untuk memahami sentimen pengguna, menemukan pola keluhan yang paling sering muncul, dan menerjemahkan hasil analisis menjadi insight bisnis yang mudah dipresentasikan.

Project ini dirancang agar cocok untuk GitHub portfolio Data Scientist junior: strukturnya rapi, alurnya end-to-end, ada notebook untuk walkthrough, ada model machine learning yang dibandingkan secara fair, dan ada aplikasi Streamlit sederhana untuk demo interaktif.

## Problem Statement

Ulasan pengguna di Google Play atau App Store sering berisi sinyal penting tentang pengalaman pelanggan. Tantangannya adalah jumlah review yang besar membuat analisis manual tidak efisien. Karena itu, project ini menjawab tiga pertanyaan utama:

1. Bagaimana distribusi sentimen pengguna terhadap aplikasi Gojek?
2. Keluhan apa yang paling sering muncul pada review negatif?
3. Model machine learning mana yang paling baik untuk mengklasifikasikan sentimen review berbahasa Indonesia?

## Dataset

Dataset yang digunakan mengacu pada referensi Kaggle notebook:

- Notebook: `najwaputrif/sentiment-analysis-review-aplikasi-gojek`
- Judul notebook: `Sentiment Analysis Review Aplikasi Gojek`

Catatan penting:

- Struktur kolom CSV pada sumber Kaggle dapat berbeda-beda tergantung file yang diunduh.
- Project ini sudah dibuat fleksibel untuk mendeteksi kolom review, rating, dan sentiment secara otomatis jika penamaannya masih umum seperti `content`, `review`, `ulasan`, `rating`, atau `sentiment`.
- Jika dataset hanya memiliki kolom rating, sentimen akan diinfer dengan aturan:
  - `4-5` = `positive`
  - `1-2` = `negative`
  - `3` = `neutral`
- Untuk supervised modeling, kelas `neutral` akan dikeluarkan agar fokus analisis tetap jelas pada `positive` vs `negative`.

### Cara menaruh dataset

Simpan file CSV hasil unduhan Kaggle ke folder berikut:

```text
data/raw/gojek_reviews.csv
```

Jika nama file berbeda, project tetap bisa membaca file CSV pertama di folder `data/raw/`, tetapi nama di atas adalah yang direkomendasikan.

## Tools dan Library

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

## Workflow Project

1. Load dataset review Gojek dari file CSV.
2. Lakukan data overview: jumlah data, kolom, missing value, distribusi sentimen, dan contoh review.
3. Bersihkan teks Bahasa Indonesia dengan lowercase, hapus simbol, angka, emoji, stopword, dan lakukan stemming menggunakan Sastrawi.
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

## Project Structure

```text
gojek-review-insight/
├── data/
│   ├── raw/
│   └── processed/
├── notebook/
│   ├── gojek_review_insight_walkthrough.py
│   └── gojek_review_insight_walkthrough.ipynb
├── src/
│   ├── __init__.py
│   └── sentiment_pipeline.py
├── app/
├── output/
│   ├── figures/
│   ├── models/
│   └── reports/
├── README.md
├── requirements.txt
└── streamlit_app.py
```

## Hasil yang Dihasilkan Pipeline

Setelah pipeline dijalankan, project akan mengekspor:

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

## Model Comparison

Project ini membandingkan tiga model klasifikasi dengan pipeline yang konsisten:

- `TF-IDF + Naive Bayes`
- `TF-IDF + Logistic Regression`
- `TF-IDF + Random Forest`

Aturan pemilihan model terbaik:

- metrik utama: `F1-score`
- alasan: F1-score lebih seimbang untuk menilai precision dan recall, terutama ketika distribusi kelas tidak sepenuhnya seimbang

Setelah Anda menjalankan pipeline pada dataset lokal, hasil metrik final akan otomatis tersimpan di:

```text
output/reports/metrics_summary.csv
```

## Insight Bisnis yang Diharapkan

Dengan struktur output yang dibuat, Anda bisa menjawab pertanyaan seperti:

- Keluhan utama pengguna paling banyak berkaitan dengan apa?
- Apakah review negatif lebih sering menyinggung error aplikasi, promo, pembayaran, atau pengalaman login?
- Aspek apa yang paling sering dipuji pengguna?
- Rekomendasi perbaikan produk apa yang paling masuk akal berdasarkan kata-kata dominan di review negatif?

File yang paling berguna untuk menjawab itu:

- `output/reports/business_insights.md`
- `output/reports/top_terms.csv`
- `output/reports/error_analysis_examples.csv`
- `output/figures/wordcloud_negative.png`

## Notebook Utama

Notebook utama tersedia dalam dua format:

- `notebook/gojek_review_insight_walkthrough.py`
- `notebook/gojek_review_insight_walkthrough.ipynb`

Versi `.py` memakai format `# %%` sehingga nyaman dibuka di VS Code dan dijalankan seperti notebook, sementara versi `.ipynb` cocok untuk presentasi langsung di Jupyter atau Colab.

## Cara Menjalankan Project

### 1. Masuk ke folder project

```powershell
cd "Gojek-Review-Insight"
```

### 2. Install dependencies

```powershell
pip install -r requirements.txt
```

### 3. Letakkan dataset CSV

Taruh file CSV di:

```text
data/raw/gojek_reviews.csv
```

### 4. Jalankan pipeline analisis

```powershell
python -m src.sentiment_pipeline
```

### 5. Jalankan aplikasi Streamlit

```powershell
streamlit run streamlit_app.py
```

## Screenshot Visualisasi

Screenshot visualisasi akan tersedia setelah pipeline dijalankan:

- `output/figures/sentiment_distribution.png`
- `output/figures/top_terms.png`
- `output/figures/wordcloud_positive.png`
- `output/figures/wordcloud_negative.png`

Anda bisa menambahkan hasil screenshot tersebut ke README ketika project sudah dieksekusi penuh pada dataset lokal.

## Kenapa Project Ini Kuat untuk Portfolio

- Topiknya relevan dan mudah dipahami recruiter
- Fokus pada Bahasa Indonesia, jadi terasa lebih kontekstual
- Mencakup EDA, preprocessing NLP, feature extraction, model comparison, evaluasi, dan business insight
- Ada error analysis, bukan hanya mengejar accuracy
- Ada app Streamlit sederhana untuk demo
- Struktur repository sudah siap untuk GitHub showcase

## Pengembangan Lanjutan

- Tambahkan normalisasi slang Bahasa Indonesia yang lebih lengkap
- Coba model lain seperti Linear SVM atau IndoBERT
- Tambahkan topic modeling untuk eksplorasi tema keluhan
- Hubungkan output ke dashboard Power BI
- Tambahkan deployment online untuk app Streamlit

## Referensi

- Kaggle notebook: `najwaputrif/sentiment-analysis-review-aplikasi-gojek`
- Repository target: `muhfajri24/Gojek-Review-Insight`

