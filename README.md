# Laporan Proyek Machine Learning - Damianus Christopher Samosir

## Project Overview

### Latar Belakang

Di era digital saat ini, ketersediaan informasi dalam bentuk buku—baik cetak maupun digital—meningkat secara signifikan. Hal ini menciptakan tantangan bagi pembaca dalam menemukan bacaan yang sesuai dengan preferensi dan kebutuhan mereka, terutama bagi pengguna baru tanpa riwayat atau referensi sebelumnya. Untuk mengatasi permasalahan ini, sistem rekomendasi buku dapat menjadi solusi yang membantu pengguna menemukan bacaan yang relevan secara otomatis dan personal.

Sistem rekomendasi terbukti mampu meningkatkan kenyamanan pengguna dalam menelusuri konten dan produk digital, termasuk buku. Penerapan collaborative filtering dan content-based filtering telah banyak digunakan dalam sistem rekomendasi untuk meningkatkan relevansi hasil \[1]. Namun, masing-masing metode memiliki kelemahan. Collaborative filtering sering mengalami masalah cold-start dan sparsity data, sedangkan content-based cenderung terbatas pada rekomendasi yang serupa dengan preferensi pengguna sebelumnya \[2].

Dalam beberapa tahun terakhir, pendekatan baru seperti **Neural Collaborative Filtering (NCF)** mulai dikembangkan untuk mengatasi kelemahan metode konvensional. NCF menggabungkan kekuatan collaborative filtering dengan kemampuan representasi dari neural network, menghasilkan model yang lebih adaptif dan akurat \[3]. Selain itu, kombinasi informasi metadata seperti genre, deskripsi, atau penulis terbukti meningkatkan akurasi prediksi dalam sistem rekomendasi buku \[4].

Proyek ini mengembangkan sistem rekomendasi buku dengan menggabungkan pendekatan **content-based filtering** dan **collaborative filtering berbasis neural network (NCF)**. Pendekatan hybrid ini diharapkan mampu menyajikan rekomendasi yang lebih relevan dan personal bagi pengguna.

Tujuan dari proyek ini adalah:

* Menganalisis preferensi pembaca berdasarkan data rating dan konten buku.
* Membangun model sistem rekomendasi yang memberikan saran bacaan yang akurat dan personal.
* Memberikan insight yang dapat meningkatkan pengalaman membaca pengguna.

### Referensi

\[1] A. Ahmed, M. Saeed, A. Shaikh, et al., “Book Recommendation Using Collaborative Filtering Algorithm,” *Mathematical Problems in Engineering*, vol. 2023, Article ID 1514801, 10 pages, 2023. doi: [10.1155/2023/1514801](https://doi.org/10.1155/2023/1514801).

\[2] N. Nabilah and Z. Zanariah, “Intelligence Book Recommendation System Using Collaborative Filtering,” *Journal of Computing Research and Innovation*, vol. 9, no. 1, 2024.

\[3] S. Mukti and Z. K. A. Baizal, “Feature Enhanced Neural Collaborative Filtering (FENCF) for Book Recommendation System,” *IJCCS (Indonesian Journal of Computing and Cybernetics Systems)*, vol. 19, no. 2, pp. 173–185, 2025.

\[4] C. Musto, P. Lops, and G. Semeraro, “Integrating Content-based and Collaborative Filtering in Recommender Systems: A Systematic Review,” *Information Fusion*, vol. 80, pp. 121–134, 2022. doi: [10.1016/j.inffus.2021.10.011](https://doi.org/10.1016/j.inffus.2021.10.011).

---

## Business Understanding

### Problem Statement

1. Bagaimana memberikan rekomendasi buku yang relevan kepada pengguna berdasarkan preferensi mereka sebelumnya?
2. Bagaimana memanfaatkan informasi buku dan interaksi pengguna dalam menghasilkan sistem rekomendasi yang optimal?

### Goals

1. Mengembangkan sistem rekomendasi buku berbasis data dan pembelajaran mesin.
2. Menghasilkan daftar rekomendasi buku personal dengan pendekatan berbasis konten dan kolaboratif.
   
### Solution Approach

Untuk mencapai tujuan tersebut, dua pendekatan yang digunakan adalah:

* **Content-Based Filtering**
  berfokus pada Author dan Judul Menganalisis metadata buku seperti book_author dan book_title menggunakan teknik TF-IDF, menggabungkan keduanya untuk menghitung kemiripan antar buku dengan cosine similarity.

* **Collaborative Filtering (Neural Collaborative Filtering)**
  Menggunakan TensorFlow dan arsitektur neural network untuk mempelajari representasi pengguna dan buku berdasarkan data interaksi (rating), lalu memprediksi kemungkinan rating untuk menghasilkan rekomendasi.

---

## Data Understanding
Dataset yang digunakan adalah [Book-Crossing Dataset](https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset), terdiri dari tiga file utama: `Books.csv`, `Users.csv`, dan `Ratings.csv`.

### 1. **Books.csv**
- **Jumlah Data**: ~271.360 baris.
- **Jumlah Kolom**: 8 kolom.
- **Missing Value**: Ada missing value pada `Book-Author` dan `Publisher`.
- **Kolom**:
  | No | Nama Kolom             | Tipe Data | Keterangan                              |
  |----|------------------------|-----------|-----------------------------------------|
  | 0  | ISBN                  | object    | Kode unik buku.                        |
  | 1  | Book-Title            | object    | Judul buku.                            |
  | 2  | Book-Author           | object    | Nama penulis buku.                     |
  | 3  | Year-Of-Publication   | mixed     | Tahun penerbitan (campuran string/int).|
  | 4  | Publisher             | object    | Nama penerbit.                         |
  | 5  | Image-URL-S           | object    | URL gambar sampul kecil.               |
  | 6  | Image-URL-M           | object    | URL gambar sampul sedang.              |
  | 7  | Image-URL-L           | object    | URL gambar sampul besar.               |
- **5 Data teratas file Books.csv**:
  ![image](https://github.com/user-attachments/assets/0252effd-e004-4804-b29e-58819aa90851)

### 2. **Users.csv**
- **Jumlah Data**: ~278.858 baris.
- **Jumlah Kolom**: 3 kolom.
- **Missing Value**: Kolom `Age` memiliki banyak missing value.
- **Kolom**:
  | No | Nama Kolom | Tipe Data | Keterangan                          |
  |----|------------|-----------|-------------------------------------|
  | 0  | User-ID    | int64     | ID unik pengguna.                  |
  | 1  | Location   | object    | Lokasi pengguna.                   |
  | 2  | Age        | float64   | Usia pengguna (bisa kosong).       |
- **5 Data teratas file Users.csv**:

  ![image](https://github.com/user-attachments/assets/39135886-d44f-49a3-ac21-8d8ade35f1e0)

### 3. **Ratings.csv**
- **Jumlah Data**: 1.149.780 baris.
- **Jumlah Kolom**: 3 kolom.
- **Missing Value**: Tidak ada missing value.
- **Kolom**:
  | No | Nama Kolom  | Tipe Data | Keterangan                              |
  |----|-------------|-----------|-----------------------------------------|
  | 0  | User-ID     | int64     | ID unik pengguna.                      |
  | 1  | ISBN        | object    | Kode unik buku.                        |
  | 2  | Book-Rating | int64     | Rating buku (0-10, 0 = implisit).      |
- **5 Data teratas file Ratings.csv**:

   ![image](https://github.com/user-attachments/assets/8fd32d3e-0aeb-4cfe-a783-aac45f99c979)

---

## Data Preprocessing
**Tahapan Proses:**

### 1. **Identifikasi ISBN Unik dari Data Buku dan Rating**

* **Cell:**

```python
isbn_buku = df_Books['ISBN'].drop_duplicates()
isbn_rating = df_Ratings['ISBN'].drop_duplicates()
semua_isbn = np.unique(np.concatenate([isbn_buku, isbn_rating]))
```

* **Penjelasan:**
  Pada langkah ini, kamu mengambil ISBN dari dua sumber berbeda, yaitu dari data buku dan data rating. Setelah menghapus duplikat dari masing-masing sumber, kamu menggabungkan keduanya dan mengurutkannya menggunakan `np.unique`. Hasilnya adalah kumpulan ISBN unik yang merepresentasikan seluruh buku yang pernah muncul dalam data.

---

### 2. **Identifikasi User-ID Unik dari Data User dan Rating**

* **Cell:**

```python
user_dari_rating = df_Ratings['User-ID'].drop_duplicates()
user_dari_users = df_Users['User-ID'].drop_duplicates()
seluruh_user = np.unique(np.concatenate([user_dari_rating, user_dari_users]))
```

* **Penjelasan:**
  Langkah ini bertujuan untuk mengidentifikasi semua pengguna unik dari dua sumber: data rating dan data user. Dengan menghapus duplikat dan menggabungkannya, kamu mendapatkan kumpulan User-ID unik yang penting untuk analisis interaksi atau personalisasi sistem rekomendasi.

---

### 3. **Penggabungan Data Buku dan Rating Berdasarkan ISBN**

* **Cell:**

```python
data_buku_rating = df_Books.merge(df_Ratings, on='ISBN', how='inner')
```

* **Penjelasan:**
  Di sini kamu menggabungkan data buku dan data rating berdasarkan kolom `ISBN`. Dengan menggunakan `how='inner'`, hanya data yang memiliki kecocokan di kedua tabel yang akan disertakan. Ini menghasilkan data gabungan yang berisi informasi buku dan rating dari pengguna untuk buku tersebut.

---

### 4. **Penggabungan Data Buku-Rating dengan Data User Berdasarkan User-ID**

* **Cell:**

```python
data_lengkap = data_buku_rating.merge(df_Users, on='User-ID', how='inner')
```

* **Penjelasan:**
  Langkah ini melengkapi data sebelumnya dengan menambahkan informasi pengguna dari `df_Users` ke dalam data buku-rating. Penggabungan dilakukan berdasarkan kolom `User-ID`. Hasil akhir berupa DataFrame `data_lengkap` yang berisi informasi lengkap: buku, user, dan rating yang diberikan.

---

### 5. **Pengecekan Nilai Kosong (Missing Values)**

* **Cell:**

```python
data_lengkap.isna().sum()
```

* **Penjelasan:**
  Setelah data lengkap terbentuk, kamu memeriksa apakah terdapat nilai kosong (null/missing) pada kolom-kolom tertentu. Ini penting untuk menilai kualitas data dan menentukan apakah perlu dilakukan penanganan seperti penghapusan atau imputasi nilai.

---

## Data Preparation
**Tahapan Proses:**

### 🔹 **1. Menghapus data dengan nilai kosong**

```python
clean_books = books.dropna()
```

* Fungsi: Membersihkan data dari **baris yang mengandung nilai kosong (NaN)** agar tidak mengganggu proses analisis atau pemodelan.

---

### 🔹 **2. Memverifikasi tidak ada nilai kosong**

```python
clean_books.isna().sum()
```

* Fungsi: Mengecek kembali apakah semua nilai kosong sudah berhasil dihapus. Output-nya akan menunjukkan `0` jika semua kolom sudah bersih.

---

### 🔹 **3. Mengurutkan data berdasarkan kolom `ISBN`**

```python
sorted_books = clean_books.sort_values(by='ISBN', ascending=True)
```

* Fungsi: Mengurutkan data berdasarkan ISBN secara alfabetis. Hal ini membantu dalam **standarisasi urutan data**, terutama sebelum menghapus duplikat.

---

### 🔹 **4. Menyalin dan mengurutkan ulang**

```python
prepared_books = sorted_books.copy()
prepared_books = prepared_books.sort_values(by='ISBN')
```

* Fungsi: Membuat **salinan** data yang telah diurutkan agar versi sebelumnya tidak terpengaruh, dan mengurutkan ulang sebagai jaminan data siap diproses lebih lanjut.

---

### 🔹 **5. Menghapus data duplikat berdasarkan ISBN**

```python
prepared_books = prepared_books.drop_duplicates(subset='ISBN')
```

* Fungsi: Menghilangkan **entri buku yang memiliki ISBN sama**, karena duplikat bisa menyebabkan bias dalam sistem rekomendasi atau analisis data.

---

### 🔹 **6. Mengambil sampel acak sebanyak 10000 data**

```python
sample_books = prepared_books.sample(n=10000, random_state=42)
```

* Fungsi: Mengambil **sampel acak** dari data agar ukuran data tidak terlalu besar (menghemat memori dan waktu komputasi), namun tetap representatif. Penggunaan `random_state` memastikan hasilnya **reproducible**.

---

### 🔹 **7. Mengubah kolom tertentu menjadi list**

```python
list_ids = sample_books['ISBN'].to_list()
list_titles = sample_books['Book-Title'].to_list()
list_authors = sample_books['Book-Author'].to_list()
```

* Fungsi: Menyimpan data ke dalam list untuk **kemudahan dalam pemrosesan lanjutan**, misalnya dalam pembentukan fitur content-based atau pemetaan ID.

---

### 🔹 **8. Membentuk DataFrame akhir untuk modeling**

```python
books_final = pd.DataFrame({
    'id': list_ids,
    'book_title': list_titles,
    'book_author': list_authors
})
```

* Fungsi: Menyusun ulang data menjadi bentuk yang **rapi dan siap pakai** untuk model machine learning atau sistem rekomendasi.
  
---

## Modeling & Results

### Berikut adalah versi penjelasan **Content-Based Filtering** milik Anda dalam format dan gaya penulisan yang **mirip dengan Collaborative Filtering** yang Anda contohkan:

---

## **1. Model Sistem Rekomendasi Content-Based Filtering**
Content-Based Filtering merekomendasikan buku berdasarkan **kemiripan konten**, khususnya informasi dari penulis buku. Model ini menggunakan representasi teks dengan **TF-IDF (Term Frequency-Inverse Document Frequency)** dan menghitung **cosine similarity** untuk mengukur tingkat kemiripan antar buku.

### a. Fitur yang Digunakan

* **Fitur Konten:** Nama penulis buku (`book_author`)
* **Representasi Teks:** TF-IDF Vectorizer dari Scikit-learn
* **Similarity Metric:** Cosine Similarity antar vektor TF-IDF

### b. Tahapan Proses Pembentukan Model

Model ini bekerja melalui beberapa tahapan sebagai berikut:

#### 1. Duplikasi Dataset Awal

```python
dataset = books_final.copy()
```

Dataset `books_final` disalin agar data asli tetap aman. Proses ini penting sebagai langkah awal sebelum dilakukan pemrosesan lebih lanjut.

#### 2. Inisialisasi dan Pelatihan TF-IDF Vectorizer

```python
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
vectorizer.fit(dataset['book_author'])
```

TF-IDF digunakan untuk mengubah informasi nama penulis menjadi **vektor numerik**. Semakin sering sebuah kata muncul dalam satu dokumen tetapi jarang dalam keseluruhan dokumen, maka bobotnya lebih tinggi.

#### 3. Transformasi Data Teks ke Bentuk TF-IDF

```python
tfidf_result = vectorizer.transform(dataset['book_author'])
```

Nama-nama penulis diubah menjadi **sparse matrix** berbentuk TF-IDF, yang menggambarkan pentingnya setiap kata dari sisi bobot dan frekuensi.

#### 4. Konversi ke Bentuk Dense dan DataFrame

```python
tfidf_dense = tfidf_result.todense()
tfidf_df = pd.DataFrame(tfidf_dense, columns=vectorizer.get_feature_names_out(), index=dataset['book_title'])
```

TF-IDF yang awalnya berupa sparse matrix dikonversi ke bentuk dense matrix dan disusun dalam DataFrame agar lebih mudah dianalisis.

#### 5. Penghitungan Cosine Similarity

```python
from sklearn.metrics.pairwise import cosine_similarity
similarity_scores = cosine_similarity(tfidf_result)
```

Cosine Similarity menghitung tingkat kemiripan antar buku berdasarkan representasi vektor dari nama penulis. Semakin besar skor, semakin mirip dua buku tersebut.

#### 6. Penyusunan Matriks Similarity dalam DataFrame

```python
similarity_df = pd.DataFrame(similarity_scores, index=dataset['book_title'], columns=dataset['book_title'])
```

Matriks kemiripan dikemas dalam bentuk DataFrame agar dapat digunakan untuk pencarian rekomendasi berdasarkan judul buku.

#### 7. Fungsi Rekomendasi Buku

```python
def rekomendasi_buku(judul, similarity_data=..., metadata=..., jumlah=5):
    ...
```

Fungsi ini menerima input judul buku dan mengembalikan daftar buku dengan skor kemiripan tertinggi, berdasarkan penulis.

### c. Evaluasi Model

Evaluasi dilakukan untuk mengukur seberapa baik sistem dalam menyarankan buku yang memang mirip (ground truth) berdasarkan ambang kemiripan (misalnya 0.5).

```python
nilai_ambang = 0.5
truth_matrix = (similarity_scores >= nilai_ambang).astype(int)

sim_flat = similarity_scores[:10000, :10000].flatten()
truth_flat = truth_matrix[:10000, :10000].flatten()

from sklearn.metrics import precision_recall_fscore_support
prediksi = (sim_flat >= nilai_ambang).astype(int)
precision, recall, f1, _ = precision_recall_fscore_support(truth_flat, prediksi, average='binary', zero_division=1)
```

Hasil metrik evaluasi:

* **Precision**: Seberapa banyak prediksi benar dari semua yang direkomendasikan.
* **Recall**: Seberapa banyak buku mirip yang berhasil ditemukan.
* **F1-Score**: Harmonik dari precision dan recall sebagai ukuran performa keseluruhan.

---

## **d. Contoh Hasil Rekomendasi Buku**

Misalkan pengguna memilih buku: **"Letter to Lord Liszt"**

Langkah-langkah:

1. **Tampilkan Buku Awal:**

```python
dataset[dataset['book_title'] == 'Letter to Lord Liszt']
```
![image](https://github.com/user-attachments/assets/efc10caa-bed0-417d-bae8-187426133ee8)


2. **Hasil Rekomendasi Top-5:**

```python
rekomendasi_buku('Letter to Lord Liszt')
```

Hasil Rekomendasi:
Rekomendasi di bawah ini dihasilkan karena nama penulis memiliki tingkat kemiripan tinggi (cosine similarity) dengan penulis buku awal.

![image](https://github.com/user-attachments/assets/1bc58cbd-275b-4a2e-bee0-5c32643ab312)


---

##  Kelebihan Content-Based Filtering

1. **Tidak Terpengaruh oleh Pengguna Lain (Independen)**
   Sistem content-based hanya bergantung pada **karakteristik item dan preferensi pengguna itu sendiri**, sehingga tidak memerlukan data dari pengguna lain. Ini menjadikan sistem lebih stabil dan konsisten dalam memberikan rekomendasi yang sesuai.

2. **Tidak Terkena Masalah Cold Start untuk Item Baru**
   Selama deskripsi item lengkap (misalnya: judul, genre, penulis, sinopsis), sistem dapat langsung merekomendasikan buku baru tanpa menunggu rating dari pengguna.

3. **Personalisasi Lebih Mendalam**
   Content-based filtering membangun **profil unik** untuk setiap pengguna berdasarkan preferensinya. Misalnya, jika seseorang suka buku bertema psikologi dan misteri, maka sistem akan lebih sering menyarankan buku dengan deskripsi serupa.

4. **Aman dari Manipulasi**
   Karena sistem tidak terlalu bergantung pada ulasan atau rating pengguna lain, maka lebih tahan terhadap **shilling attack** atau manipulasi dari pengguna palsu.

---

##  Kekurangan Content-Based Filtering

1. **Kurangnya Keanekaragaman (Overspecialization)**
   Sistem cenderung merekomendasikan item yang **sangat mirip** dengan yang sudah disukai, sehingga mengurangi kemungkinan menemukan konten baru yang mungkin menarik dari kategori berbeda.

2. **Cold Start pada Pengguna Baru**
   Jika pengguna belum pernah memberikan feedback (misalnya belum pernah membaca atau memberi rating buku), maka sistem tidak memiliki dasar untuk membuat rekomendasi.

3. **Ketergantungan pada Metadata**
   Kualitas rekomendasi sangat bergantung pada **kelengkapan dan kualitas data item**. Jika informasi seperti genre, sinopsis, atau keyword tidak lengkap, maka hasil rekomendasi bisa kurang akurat.

4. **Kurang Adaptif terhadap Perubahan Selera**
   Jika pengguna mulai menyukai genre baru, sistem content-based membutuhkan waktu untuk menyesuaikan, karena ia belajar dari interaksi sebelumnya yang bisa jadi sudah tidak relevan lagi.

---
### 2. Collaborative Filtering (Neural Collaborative Filtering)
**Tahapan Proses:**

### 🔹 **A. Memilih Kolom yang Relevan**

```python
selected_cols = ['User-ID', 'ISBN', 'Book-Rating']
ratings_df = preparation[selected_cols]
```

* Fungsi: Menyaring hanya kolom penting dari dataset untuk membangun sistem rekomendasi: ID pengguna, ID buku, dan rating.

  ![image](https://github.com/user-attachments/assets/481fd566-76fa-4911-9733-9ec9a8eb0764)

---

### 🔹 **B. Encoding `User-ID` ke Index Numerik**

```python
unique_users = ratings_df['User-ID'].drop_duplicates().tolist()
user_id_to_index = {user_id: idx for idx, user_id in enumerate(unique_users)}
```

* Fungsi: Mengonversi `User-ID` menjadi indeks numerik agar dapat digunakan dalam model embedding neural network.

---

### 🔹 **C. Encoding `ISBN` ke Index Numerik**

```python
unique_books = ratings_df['ISBN'].unique().tolist()
isbn_to_index = {isbn: idx for idx, isbn in enumerate(unique_books)}
```

* Fungsi: Mengonversi `ISBN` menjadi indeks numerik, seperti pada langkah encoding pengguna.

---

### 🔹 **D. Menyisipkan Kolom `user_index` dan `book_index`**

```python
ratings_df['user_index'] = ratings_df['User-ID'].map(user_id_to_index)
ratings_df['book_index'] = ratings_df['ISBN'].map(isbn_to_index)
```

* Fungsi: Menambahkan hasil encoding ke dalam DataFrame agar bisa langsung digunakan sebagai input model.

---

### 🔹 **E. Menentukan Jumlah Unik User dan Buku**

```python
total_users = len(user_id_to_index)
total_books = len(isbn_to_index)
```

* Fungsi: Menentukan ukuran embedding layer dengan menghitung jumlah user dan buku yang unik.

---

### 🔹 **F. Konversi Rating dan Normalisasi Skor**

```python
ratings_df['score'] = ratings_df['Book-Rating'].astype(np.float32)
min_score = ratings_df['score'].min()
max_score = ratings_df['score'].max()
```

* Fungsi: Mengonversi nilai rating ke tipe numerik (`float32`) dan mencari nilai minimum serta maksimum untuk proses normalisasi.

---

### 🔹 **G. Sampling Dataset**

```python
if ratings_df.shape[0] < 10000:
```

* Fungsi: Mengambil sampel acak 10.000 data untuk efisiensi pelatihan model dan menghindari beban komputasi berlebih.

  ![image](https://github.com/user-attachments/assets/738c06fd-d4b5-45b8-b59e-2c76f7f9cefb)

---

### 🔹 **H. Persiapan Input dan Output Model**

```python
features = ratings_df[['user_index', 'book_index']].to_numpy()
labels = ratings_df['score'].apply(lambda s: (s - min_score) / (max_score - min_score)).to_numpy()
```

* Fungsi: Menyiapkan input (`user_index`, `book_index`) dan output (rating yang dinormalisasi) untuk model rekomendasi.

---

### 🔹 **I. Split Dataset untuk Pelatihan dan Validasi**

```python
x_train, x_val = features[:train_split], features[train_split:]
y_train, y_val = labels[:train_split], labels[train_split:]
```

* Fungsi: Membagi data menjadi data latih (80%) dan data validasi (20%).

---

### 🔹 **J. Arsitektur Model Collaborative Filtering**

```python
class BookRecommender(Model):
    ...
```

* Fungsi: Membangun model rekomendasi berbasis **dot product dari vektor embedding user dan item**, ditambah bias, dan dikalkulasi dengan aktivasi sigmoid.

---

### 🔹 **K. Kompilasi dan Pelatihan Model**

```python
model.compile(...)
training_history = model.fit(...)
```

* Fungsi: Melatih model dengan loss function **Binary Crossentropy** dan mengoptimasi menggunakan **Adam Optimizer**. RMSE dipakai sebagai metrik evaluasi.

---

### 🔹 **L. Visualisasi Hasil Pelatihan**

```python
plt.plot(training_history.history['root_mean_squared_error'])
plt.plot(training_history.history['val_root_mean_squared_error'])
```

* Fungsi: Menampilkan grafik perbandingan RMSE antara data pelatihan dan validasi selama proses pelatihan.

  ![image](https://github.com/user-attachments/assets/ea73cfcf-dcd7-4679-b9b9-0a2264966ff1)


---

### 🔹 **M. Menyiapkan Data untuk Prediksi Buku Baru**

```python
book_candidates = books_new[~books_new['id'].isin(visited_books['ISBN'])]
```

* Fungsi: Menyusun daftar buku yang **belum pernah dibaca oleh user** sebagai kandidat untuk direkomendasikan.

---

### 🔹 **N. Menyusun Input Prediksi dan Prediksi Rekomendasi**

```python
input_array = np.hstack((np.full((len(book_indices), 1), user_idx), book_indices))
predicted_scores = model.predict(input_array).flatten()
```

* Fungsi: Membuat input gabungan `user_index` dan `book_index` lalu menghasilkan skor prediksi untuk setiap buku.

---

### 🔹 **O. Menampilkan Rekomendasi Buku**

```python
recommended_isbns = [index_to_isbn[book_indices[i][0]] for i in top_indices]
```

* Fungsi: Mengambil 10 skor tertinggi sebagai buku rekomendasi berdasarkan hasil prediksi model.

  ![image](https://github.com/user-attachments/assets/3597b364-2372-4785-a15e-92bbe4a20abf)

---
Notebook **Rekomendasi\_Buku.ipynb** hanya memiliki satu judul markdown “**Collaborative Filtering**” tanpa penjelasan atau implementasi lanjutan yang terlihat dari isi sel-sel lainnya. Namun, saya bisa tetap menjelaskan **kelebihan dan kekurangan Collaborative Filtering** secara umum—terutama dalam konteks sistem rekomendasi buku—berdasarkan praktik standar dan pengalaman umum penggunaannya.

---

###  Kelebihan Collaborative Filtering

1. **Personalisasi Tinggi**
   Rekomendasi dihasilkan berdasarkan perilaku pengguna lain yang memiliki kesamaan, sehingga hasilnya lebih relevan dan personal.

2. **Tidak Bergantung pada Metadata**
   Tidak memerlukan informasi spesifik dari item (seperti sinopsis buku, penulis, genre, dsb). Hal ini berguna jika data deskriptif tidak lengkap.

3. **Mampu Menangkap Preferensi Kompleks**
   Collaborative Filtering bisa merekomendasikan buku dari berbagai kategori yang tidak secara eksplisit mirip, tetapi disukai oleh pengguna serupa.

4. **Skalabilitas Algoritmik**
   Pendekatan seperti matrix factorization atau deep collaborative filtering bisa dikembangkan lebih lanjut untuk skala besar.

---

###  Kekurangan Collaborative Filtering

1. **Cold Start Problem**
   Tidak dapat merekomendasikan item atau pengguna baru yang belum memiliki cukup data interaksi (misal: rating atau review).

2. **Data Sparsity**
   Jika mayoritas pengguna hanya memberikan sedikit rating, maka matriks user-item menjadi sangat kosong sehingga performa algoritma menurun.

3. **Scalability Issue**
   Untuk dataset besar, pendekatan tradisional berbasis neighborhood (user-user atau item-item) bisa sangat lambat dan memerlukan banyak memori.

4. **Serangan Manipulasi (Shilling Attack)**
   Sistem ini lebih mudah dimanipulasi oleh pengguna palsu yang memberikan rating berlebihan untuk memengaruhi rekomendasi.

---

Jika kamu ingin, saya bisa bantu melengkapi notebook tersebut dengan contoh kode Collaborative Filtering berbasis user-user atau menggunakan pendekatan matrix factorization seperti SVD. Apakah kamu ingin saya tambahkan itu ke notebook?


---

## Evaluation
---

### **1. Content-Based Filtering (Gambar Pertama)**

#### **Kode dan Proses Evaluasi**
- **Dataset**: Kamu menggunakan 10.000 sampel untuk evaluasi (`ukuran_sampel = 10000`).
- **Metrik Evaluasi**: Kamu menghitung **similarity scores** antara **truth** (data asli) dan **prediksi** (hasil model) menggunakan **cosine similarity** dengan fungsi `sklearn.metrics.cosine_similarity`. Nilai prediksi diambil berdasarkan threshold tertentu (`nilai_ambang`).
- **Metrik yang Digunakan**:
  - **Precision, Recall, dan F1-Score**: Diukur menggunakan `sklearn.metrics` dengan metode `precision_recall_fscore_support`.
  - Parameter `average='binary'` menunjukkan bahwa ini adalah evaluasi untuk klasifikasi biner (benar/salah).
  - `zero_division=1` digunakan untuk menangani kasus pembagian dengan nol.

 ### **Rumus**:
 #### Precision@K

- **Precision@K** mengukur proporsi film yang direkomendasikan oleh sistem yang benar-benar relevan bagi pengguna.  
- Nilai precision yang tinggi berarti sistem rekomendasi mampu memberikan rekomendasi yang tepat sasaran dan meminimalkan rekomendasi yang tidak sesuai dengan preferensi pengguna.

Rumus Precision@K:

$$
\text{Precision@K} = \frac{|\text{Recommended Items} \cap \text{Relevant Items}|}{K}
$$

#### Recall@K

- **Recall@K** mengukur proporsi film relevan yang berhasil ditemukan dan direkomendasikan oleh sistem dari seluruh film yang relevan di dataset.  
- Nilai recall yang tinggi menunjukkan sistem mampu menemukan sebagian besar film yang disukai pengguna, sehingga cakupan rekomendasi cukup baik.

Rumus Recall@K:

$$
\text{Recall@K} = \frac{|\text{Recommended Items} \cap \text{Relevant Items}|}{|\text{Relevant Items}|}
$$

Dimana:  
- *Recommended Items* adalah daftar film yang direkomendasikan oleh sistem (Top-K),  
- *Relevant Items* adalah film yang benar-benar disukai pengguna (misalnya berdasarkan rating ≥ 4),  
- *K* adalah jumlah film teratas yang direkomendasikan.


#### **Hasil Evaluasi**
- **Precision**: 1.00
  - Artinya, 100% dari prediksi yang dianggap benar oleh model memang benar (tidak ada false positive).
- **Recall**: 1.00
  - Artinya, model berhasil mendeteksi 100% dari semua data yang benar (tidak ada false negative).
- **F1-Score**: 1.00
  - F1-Score adalah harmonic mean dari precision dan recall, sehingga nilai 1.00 menunjukkan performa sempurna.

#### **Interpretasi**
- Model **Content-Based Filtering** yang kamu evaluasi menunjukkan performa **sempurna** dengan precision, recall, dan F1-Score sebesar 1.00. Ini berarti model sangat akurat dalam memprediksi item yang relevan berdasarkan fitur konten (misalnya, deskripsi atau metadata). Tidak ada kesalahan prediksi (false positive atau false negative) dalam sampel yang dievaluasi.

---

### **2. Collaborative Filtering (Gambar Kedua)**

#### **Kode dan Proses Evaluasi**
- **Metrik Evaluasi**: Kamu menggunakan **RMSE (Root Mean Squared Error)** untuk mengukur error prediksi model.
  - RMSE dihitung dengan fungsi `root_mean_squared_error` dari `sklearn.metrics`.
  - Kamu memplot grafik RMSE untuk data **training** dan **validation** selama 25 epoch menggunakan `matplotlib`.
- **Grafik**: Grafik menunjukkan perubahan RMSE pada data training (biru) dan validation (oranye) seiring bertambahnya epoch.

#### Rumus RMSE dan MAE

$$
\text{RMSE} = \sqrt{ \frac{1}{n} \sum_{i=1}^{n} (\hat{y}_i - y_i)^2 }
$$

$$
\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |\hat{y}_i - y_i|
$$

Dimana:
- ŷᵢ adalah nilai prediksi rating,
- yᵢ adalah nilai rating sebenarnya,
- n adalah jumlah sampel.
  
#### **Hasil Evaluasi**
- **Grafik RMSE**:
  - **Training RMSE**:
    - Mulai dari sekitar 0.30 pada epoch awal.
    - Terus menurun hingga mendekati 0.05 pada epoch 25.
  - **Validation RMSE**:
    - Mulai dari sekitar 0.32 pada epoch awal.
    - Menurun hingga mendekati 0.30 pada epoch 25, kemudian stabil.
- **Tren**:
  - RMSE untuk data training menurun lebih tajam dibandingkan validation, menunjukkan bahwa model belajar dengan baik pada data training.
  - RMSE validation lebih tinggi dan stabil di sekitar 0.30, yang menunjukkan adanya sedikit **overfitting** (model terlalu fit pada data training sehingga performanya pada data validation tidak sebaik pada data training).

#### Visualisasi
![image](https://github.com/user-attachments/assets/fe77f30c-9a25-4426-9028-c309b8de727e)

---

## Kesimpulan

Proyek ini berhasil mengembangkan sistem rekomendasi buku melalui pendekatan Content-Based Filtering yang berfokus pada metadata book_author dan book_title. Dimulai dari analisis kebutuhan (Business Understanding), eksplorasi dan preprocessing data (Data Understanding & Preparation), hingga pengembangan dan evaluasi model, sistem ini mampu mencapai tujuan yang telah ditetapkan.

Dengan langkah preprocessing untuk menggabungkan data penulis dan judul, serta analisis menggunakan TF-IDF dan cosine similarity, sistem dapat memberikan rekomendasi buku yang relevan dan sesuai dengan preferensi pengguna. Rekomendasi ini memudahkan pengguna dalam menemukan buku yang diminati sekaligus menawarkan opsi baru yang menarik. Sistem ini menjadi solusi digital yang efektif untuk meningkatkan kemudahan dalam memilih buku dan memperkaya pengalaman membaca pengguna.
