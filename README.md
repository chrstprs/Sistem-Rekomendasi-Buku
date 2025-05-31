# Sistem-Rekomendasi-Buku

# Laporan Proyek Machine Learning - [Nama Anda]

## Project Overview

Sistem rekomendasi buku merupakan alat penting dalam membantu pengguna menemukan buku yang sesuai dengan preferensi mereka, terutama di era digital dengan jumlah buku yang sangat banyak. Proyek ini bertujuan untuk membangun sistem rekomendasi buku menggunakan dataset yang berisi informasi tentang pengguna, buku, dan rating. Dengan memanfaatkan pendekatan berbasis machine learning, proyek ini berfokus untuk memberikan rekomendasi buku yang relevan kepada pengguna berdasarkan riwayat rating mereka.

**Mengapa masalah ini perlu diselesaikan?**  
Peningkatan jumlah buku yang tersedia di platform digital menyebabkan pengguna sering kali kesulitan memilih buku yang sesuai dengan minat mereka. Sistem rekomendasi membantu mengurangi *information overload* dan meningkatkan pengalaman pengguna dengan memberikan saran yang dipersonalisasi. Pendekatan ini juga dapat meningkatkan kepuasan pengguna dan potensi pembelian di platform buku.

**Referensi**  
1. J. Bobadilla, F. Ortega, A. Hernando, and A. Gutiérrez, "Recommender systems survey," *Knowledge-Based Systems*, vol. 46, pp. 109–132, Jul. 2013. [Online]. Available: https://doi.org/10.1016/j.knosys.2013.03.012
2. C. C. Aggarwal, *Recommender Systems: The Textbook*. Springer, 2016.

## Business Understanding

### Problem Statements
- **Masalah 1**: Pengguna kesulitan menemukan buku yang sesuai dengan preferensi mereka karena banyaknya pilihan buku di platform digital.
- **Masalah 2**: Kurangnya personalisasi dalam rekomendasi buku menyebabkan rendahnya kepuasan pengguna dan potensi pembelian yang terlewat.
- **Masalah 3**: Dataset rating buku mengandung nilai nol (implisit rating) yang dapat memengaruhi akurasi rekomendasi jika tidak ditangani dengan baik.

### Goals
- **Tujuan 1**: Membuat sistem rekomendasi yang dapat menyarankan buku berdasarkan preferensi pengguna dengan memanfaatkan data rating.
- **Tujuan 2**: Mengembangkan model yang memberikan rekomendasi personalisasi untuk meningkatkan pengalaman pengguna.
- **Tujuan 3**: Menangani data rating implisit (nilai nol) untuk meningkatkan akurasi prediksi model rekomendasi.

### Solution Approach
- **Pendekatan 1**: Menggunakan *Collaborative Filtering* dengan model deep learning berbasis TensorFlow untuk memprediksi rating buku berdasarkan interaksi pengguna dan buku.
- **Pendekatan 2**: Menerapkan *Content-Based Filtering* menggunakan *TF-IDF Vectorizer* dan *cosine similarity* untuk merekomendasikan buku berdasarkan kesamaan fitur seperti judul dan penulis.
- **Pendekatan 3**: Menggabungkan pendekatan *hybrid* (kombinasi *collaborative* dan *content-based filtering*) untuk meningkatkan akurasi rekomendasi dengan memanfaatkan kelebihan kedua metode.

## Data Understanding

Dataset yang digunakan dalam proyek ini terdiri dari tiga file utama: `Ratings.csv`, `Users.csv`, dan `Books.csv`. Dataset ini bersumber dari [Kaggle Book-Crossing Dataset](https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset). Berikut adalah detailnya:

- **Jumlah Data**:
  - `Ratings.csv`: 1.149.780 entri, berisi informasi rating buku oleh pengguna.
  - `Users.csv`: Berisi informasi demografis pengguna seperti ID dan lokasi.
  - `Books.csv`: Berisi metadata buku seperti ISBN, judul, penulis, dan tahun terbit.
- **Kondisi Data**: Dataset memiliki beberapa tantangan, seperti kolom dengan tipe data campuran pada `Books.csv` (kolom tahun terbit) dan rating implisit (nilai 0) yang mendominasi `Ratings.csv`.

**Variabel pada Dataset**:
- **Ratings.csv**:
  - `User-ID`: ID unik untuk setiap pengguna (integer).
  - `ISBN`: Kode unik untuk setiap buku (string).
  - `Book-Rating`: Rating yang diberikan pengguna untuk buku, dalam skala 0-10 (integer, dengan 0 menunjukkan rating implisit).
- **Users.csv**:
  - `User-ID`: ID unik pengguna (integer).
  - `Location`: Lokasi pengguna (string).
  - `Age`: Usia pengguna (integer, nullable).
- **Books.csv**:
  - `ISBN`: Kode unik buku (string).
  - `Book-Title`: Judul buku (string).
  - `Book-Author`: Nama penulis buku (string).
  - `Year-Of-Publication`: Tahun penerbitan buku (campuran string dan integer).
  - `Publisher`: Penerbit buku (string).

**Exploratory Data Analysis**:
- Distribusi rating menunjukkan bahwa rating 0 (implisit) mendominasi dengan 716.109 entri, sedangkan rating eksplisit (1-10) memiliki jumlah yang jauh lebih sedikit, dengan rating 8 sebagai yang tertinggi (103.736 entri).
- Visualisasi distribusi rating dilakukan menggunakan `df_Ratings.groupby('Book-Rating').count()` untuk memahami pola rating pengguna.

## Data Preparation

Berikut adalah langkah-langkah *data preparation* yang dilakukan:

1. **Pengunggahan Data**:
   ```python
   df_Ratings = pd.read_csv('Ratings.csv')
   df_Users = pd.read_csv('Users.csv')
   df_Books = pd.read_csv('Books.csv')
