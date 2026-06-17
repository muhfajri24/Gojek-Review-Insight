# Gojek Review Insight Summary

## Ringkasan Dataset

- Total review setelah preprocessing: `6,152`
- Distribusi sentimen: `{'positive': 3131, 'negative': 2670, 'neutral': 351}`
- Model terbaik berdasarkan F1-score: `Naive Bayes`
- F1-score terbaik: `0.8818`

## Keluhan Utama Pengguna

Kata yang paling sering muncul pada review negatif mengarah ke tema seperti: `driver, makin, gopay, mau, padahal, lama, bayar, malah`.
Secara praktis, ini biasanya berkaitan dengan masalah performa aplikasi, proses login, promo, pembayaran, atau pengalaman pemesanan yang tidak konsisten.

## Aspek yang Disukai Pengguna

Review positif lebih sering menonjolkan kata seperti: `sangat, bantu, bagus, baik, mudah, mantap, cepat, banyak`.
Ini biasanya menunjukkan apresiasi pengguna terhadap kemudahan penggunaan, kecepatan layanan, promo yang menarik, dan manfaat aplikasi dalam aktivitas harian.

## Error Analysis

- Jumlah review yang salah diprediksi model terbaik: `140`
- Pola umum kesalahan: review bercampur antara pujian dan keluhan, bahasa gaul, typo, dan konteks sarkastik.

## Rekomendasi Bisnis

1. Prioritaskan investigasi pada tema negatif yang paling dominan di `top_terms.csv` dan `wordcloud_negative.png`.
2. Audit perjalanan pengguna pada area yang paling sering dikeluhkan seperti pembayaran, promo, atau stabilitas aplikasi.
3. Pertahankan elemen yang sering dipuji pengguna dan jadikan itu bahan komunikasi produk.
4. Gunakan review salah prediksi sebagai sumber tambahan untuk memperkaya kamus normalisasi slang Bahasa Indonesia.
