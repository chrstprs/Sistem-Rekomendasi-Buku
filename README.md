# Sistem-Rekomendasi-Buku

# Laporan Proyek Machine Learning - Christ

## Project Overview

Perkembangan teknologi machine learning membuka peluang besar dalam membantu masyarakat menemukan informasi yang relevan secara otomatis, salah satunya melalui sistem rekomendasi. Sistem ini sangat bermanfaat dalam berbagai bidang, termasuk dalam dunia literasi, seperti perpustakaan digital dan platform pembaca buku online. Di tengah rendahnya tingkat literasi di Indonesia \[[1](https://www.tribunnews.com/nasional/2021/03/22/tingkat-literasi-indonesia-di-dunia-rendah-ranking-62-dari-70-negara)], sistem rekomendasi buku dapat menjadi solusi untuk mendorong minat baca masyarakat.

Proyek ini mengusung judul **Pembuatan Sistem Rekomendasi Buku Menggunakan Content-Based Filtering dan Neural Collaborative Filtering (NCF)**.

## Business Understanding

### Problem Statements

Berdasarkan latar belakang di atas, maka permasalahan yang ingin diselesaikan dalam proyek ini adalah:

* Bagaimana memberikan rekomendasi buku yang relevan kepada pengguna berdasarkan preferensi mereka?
* Bagaimana memanfaatkan data rating dan informasi konten buku secara efektif untuk membangun sistem rekomendasi?

### Goals

Tujuan dari proyek ini adalah:

* Membangun sistem rekomendasi buku yang mampu memberikan saran personal berdasarkan data konten dan rating dari pengguna.
* Menerapkan dua pendekatan utama yaitu content-based filtering dan collaborative filtering menggunakan deep learning (Neural Collaborative Filtering).

### Solution Statement

Solusi yang diterapkan adalah menggabungkan dua pendekatan:

* **Content-Based Filtering**: memberikan rekomendasi berdasarkan kesamaan konten buku, seperti kategori dan deskripsi.
* **Neural Collaborative Filtering (NCF)**: model deep learning yang mempelajari representasi pengguna dan item melalui embedding dan memprediksi rating berdasarkan interaksi historis.

Kombinasi dua metode ini memungkinkan sistem memberikan rekomendasi yang lebih personal dan akurat.

## Data Understanding

Dataset yang digunakan merupakan data hasil scraping dari situs GoodReads yang terdiri dari tiga bagian utama:

* `books.csv`: berisi informasi metadata buku seperti judul, penulis, genre, dan deskripsi.
* `ratings.csv`: berisi data rating yang diberikan oleh pengguna terhadap buku tertentu.
* `users.csv`: berisi informasi dasar pengguna (jika tersedia).

Contoh informasi penting dari data buku:

* `book_id`, `title`, `authors`, `categories`, `description`, `average_rating`

Contoh informasi dari data rating:

* `user_id`, `book_id`, `rating`

Jumlah data:

* Lebih dari 10.000 buku
* Lebih dari 1.000.000 rating

Visualisasi awal menunjukkan bahwa data rating cenderung tidak seimbang, dengan sebagian besar pengguna memberikan rating tinggi (skor 4 atau 5).

## Data Preparation

Tahapan preprocessing meliputi:

* **Pembersihan Data**: menghapus nilai null dan duplikat dari data buku dan rating.
* **Handling Imbalanced Data**: hanya rating positif (misalnya rating > 3) yang digunakan untuk training model.
* **Encoding ID**: `user_id` dan `book_id` di-encode menjadi indeks numerik untuk keperluan embedding.
* **Feature Engineering (Content-Based)**:

  * Pembuatan vektor fitur dari teks deskripsi buku dan kategori menggunakan TF-IDF.
  * Menghitung kemiripan antar buku menggunakan cosine similarity.
* **Normalisasi**: Skala nilai rating distandarisasi ke rentang 0-1.
* **Split Data**: Data dibagi menjadi training (80%) dan validation (20%).

## Modeling

### Content-Based Filtering

Model menghitung kesamaan antar buku berdasarkan konten teks. Tahapan utama:

* Vektorisasi deskripsi dan genre dengan TF-IDF
* Hitung cosine similarity antar vektor buku
* Ambil top-N buku yang mirip berdasarkan skor similarity

Model ini mampu memberikan rekomendasi meskipun pengguna belum memberikan rating (cold-start solution).

### Collaborative Filtering (Neural Collaborative Filtering)

Model NCF yang dibangun menggunakan TensorFlow terdiri dari:

* Embedding layer untuk user dan item
* Beberapa dense layer (fully connected) untuk memodelkan interaksi non-linear
* Output layer dengan aktivasi sigmoid untuk menghasilkan prediksi rating

Arsitektur model:

* Ukuran embedding: 50
* Hidden layers: 128 → 64 → 32
* Loss function: Binary Crossentropy
* Optimizer: Adam

Model ini dilatih selama beberapa epoch hingga mencapai nilai loss minimum.

Hasil prediksi digunakan untuk membuat top-N rekomendasi untuk tiap pengguna.

## Evaluation

Metrik yang digunakan untuk evaluasi adalah:

* **Root Mean Squared Error (RMSE)** untuk model NCF
* **Precision\@K dan Recall\@K** untuk mengukur seberapa tepat rekomendasi terhadap data sebenarnya

Model NCF berhasil mencapai RMSE di bawah 0.2, yang menunjukkan prediksi cukup akurat.

Untuk content-based filtering, evaluasi dilakukan secara manual dengan memeriksa hasil kemiripan dan relevansi dari rekomendasi.

## Conclusion

Proyek ini berhasil membangun sistem rekomendasi buku menggunakan dua pendekatan:

* Content-Based Filtering: efektif untuk cold-start dan berbasis konten buku.
* Neural Collaborative Filtering: menghasilkan rekomendasi yang lebih personal berdasarkan pola rating.

Sistem ini dapat dikembangkan lebih lanjut dengan:

* Menambahkan side information seperti demografi pengguna
* Menggabungkan kedua pendekatan (hybrid model)
* Menggunakan model transformer atau sequence-based recommendation

## Referensi

\[[1](https://www.tribunnews.com/nasional/2021/03/22/tingkat-literasi-indonesia-di-dunia-rendah-ranking-62-dari-70-negara)] Utami, L. D. (2021). *Tingkat Literasi Indonesia di Dunia Rendah, Ranking 62 Dari 70 Negara*. Tribunnews.

\[[2](https://developers.google.com/machine-learning/recommendation)] Google Developers. *Recommendation Systems: Collaborative Filtering*.

\[[3](https://towardsdatascience.com/building-a-book-recommendation-system-using-neural-collaborative-filtering-5e5bdf3f4f65)] Towards Data Science. *Building a Book Recommendation System Using NCF*.



