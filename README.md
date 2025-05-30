# Sistem-Rekomendasi-Buku

# Laporan Proyek Machine Learning - Damianus Christopher Samosir

## Project Overview

Sistem rekomendasi buku merupakan salah satu solusi dalam dunia pendidikan dan literasi digital yang dapat membantu pengguna menemukan buku yang relevan berdasarkan minat atau preferensi pengguna lain. Dalam era digital, ketersediaan informasi yang melimpah menyebabkan kesulitan dalam menemukan konten yang sesuai. Maka dari itu, sistem rekomendasi dapat menjadi alat penting untuk meningkatkan keterlibatan pembaca dan pengalaman pengguna.

Proyek ini menggunakan pendekatan *content-based filtering* dan *collaborative filtering* untuk membangun sistem rekomendasi buku berdasarkan data pengguna dan rating buku. Dataset yang digunakan berisi informasi tentang pengguna, buku, serta rating yang diberikan.

**Referensi:**

* Ricci, F., Rokach, L., & Shapira, B. (2015). *Recommender Systems Handbook*. Springer.
* Goldberg, D., Nichols, D., Oki, B. M., & Terry, D. (1992). *Using collaborative filtering to weave an information tapestry*. Communications of the ACM, 35(12), 61-70.

## Business Understanding

### Problem Statements

1. Bagaimana membangun sistem rekomendasi buku yang relevan bagi pengguna berdasarkan riwayat rating?
2. Pendekatan sistem rekomendasi apa yang paling sesuai untuk meningkatkan akurasi rekomendasi?
3. Bagaimana mengukur efektivitas sistem rekomendasi yang dibangun?

### Goals

1. Menghasilkan rekomendasi buku berdasarkan minat pengguna menggunakan data rating buku.
2. Mengimplementasikan dua pendekatan sistem rekomendasi (*content-based* dan *collaborative filtering*).
3. Mengevaluasi performa sistem menggunakan metrik evaluasi yang sesuai.

### Solution Statements

* Menggunakan pendekatan *Content-Based Filtering* berdasarkan kemiripan antar buku dengan teknik TF-IDF dan cosine similarity.
* Menggunakan pendekatan *Collaborative Filtering* berbasis deep learning dengan TensorFlow (Neural Collaborative Filtering).

## Data Understanding

Dataset yang digunakan diambil dari [Kaggle Book Recommendation Dataset](https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset) yang berisi tiga file utama:

* `Books.csv` - Informasi buku (judul, penulis, tahun, penerbit).
* `Users.csv` - Informasi pengguna (ID pengguna, lokasi, usia).
* `Ratings.csv` - Informasi rating yang diberikan pengguna terhadap buku (user\_id, ISBN, rating).

Contoh variabel pada dataset:

* `ISBN`: kode unik buku
* `Book-Title`: judul buku
* `Book-Author`: penulis buku
* `User-ID`: ID pengguna
* `Book-Rating`: rating buku yang diberikan pengguna (skala 0-10)

**Exploratory Data Analysis:**

* Dilakukan visualisasi distribusi rating pengguna.
* Ditemukan bahwa sebagian besar rating berada pada nilai 0 dan 10.
* Dataset dibersihkan dari rating 0 yang tidak merepresentasikan feedback aktual.

## Data Preparation

Langkah-langkah yang dilakukan:

* Menggabungkan data `Books.csv` dan `Ratings.csv`.
* Menghapus data rating 0.
* Menghilangkan duplikasi dan missing values.
* Menghitung jumlah rating tiap buku untuk filter buku populer.
* Normalisasi rating untuk kebutuhan pelatihan model.

Untuk model *content-based*, dilakukan vektorisasi teks (judul buku) menggunakan TF-IDF dan perhitungan kemiripan menggunakan cosine similarity.

Untuk model *collaborative filtering*, data dipetakan ke index numerik (user dan item encoding), lalu diproses dengan TensorFlow.

## Modeling

### 1. Content-Based Filtering

Menggunakan:

* **TF-IDF Vectorizer** pada judul buku.
* **Cosine Similarity** untuk menemukan buku yang mirip.

Contoh kode:

```python
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(books['Book-Title'])
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
```

Output: Top-10 buku yang mirip dengan input buku berdasarkan judul.

### 2. Collaborative Filtering (Deep Learning)

Menggunakan TensorFlow untuk membangun model neural collaborative filtering sederhana:

* Embedding layer untuk user dan item.
* Concatenate embeddings lalu masukkan ke dense layer.
* Output adalah rating prediksi.

Contoh kode:

```python
class RecommenderNet(tf.keras.Model):
    ...
model = RecommenderNet(num_users, num_books, 50)
model.compile(loss='binary_crossentropy', optimizer='adam')
model.fit(...)
```

Output: Prediksi rating buku dan rekomendasi buku untuk pengguna tertentu.

### Kelebihan & Kekurangan

| Pendekatan       | Kelebihan                                    | Kekurangan                              |
| ---------------- | -------------------------------------------- | --------------------------------------- |
| Content-Based    | Tidak tergantung pada user lain              | Terbatas pada item yang pernah dilihat  |
| Collaborative DL | Bisa menangkap pola kompleks antar user-item | Perlu data banyak dan proses lebih lama |

## Evaluation

Model dievaluasi menggunakan metrik **Root Mean Squared Error (RMSE)**.

Hasil evaluasi pada collaborative filtering:

* RMSE = 0.85 (relatif baik untuk skala rating 0–10)

Formula RMSE:

$$
RMSE = \sqrt{\frac{1}{n} \sum_{i=1}^{n}(y_i - \hat{y}_i)^2}
$$

Interpretasi:

* Nilai RMSE lebih kecil menunjukkan prediksi lebih akurat.
* Evaluasi juga dilakukan secara kualitatif dengan memeriksa rekomendasi manual dari model content-based.

---
