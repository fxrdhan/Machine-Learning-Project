# Proyek Machine Learning
## Kombinasi Unsupervised dan Supervised Learning

### 1. DESKRIPSI PROYEK
Proyek ini merupakan implementasi kombinasi teknik machine learning yang menggabungkan unsupervised learning (clustering) dan supervised learning (classification). Pendekatan ini memungkinkan kita untuk:
1. Menemukan pola tersembunyi dalam data menggunakan clustering
2. Menggunakan hasil clustering sebagai label untuk membangun model prediktif

### 2. REQUIREMENTS PROYEK

#### 2.1 Teknologi yang Digunakan
- Bahasa Pemrograman: Python
- Format File:
  - Notebook Clustering (.ipynb)
  - Notebook Classification (.ipynb)
  - Dataset Original (.csv)
  - Dataset dengan Label Clustering (.csv)

#### 2.2 Kriteria Dataset
- Jumlah minimum baris: 2,500 records
- Jumlah minimum fitur: 5 features
- Karakteristik: Dataset tanpa label (untuk tahap awal clustering)

### 3. TAHAPAN PENGEMBANGAN

#### 3.1 Tahap Persiapan Data
1. Pengumpulan Dataset
   - Mencari dataset yang sesuai kriteria
   - Memastikan kualitas dan kuantitas data
   - Verifikasi minimum 2,500 baris dan 5 fitur

2. Pembersihan Data
   - Penanganan missing values
   - Penghapusan noise
   - Penanganan inkonsistensi data
   - Normalisasi/standardisasi (jika diperlukan)

#### 3.2 Tahap Clustering (Unsupervised Learning)

1. Feature Selection
   - Implementasi metode feature selection
   - Dokumentasi fitur yang terpilih
   - Perbandingan performa sebelum dan sesudah feature selection

2. Pembangunan Model Clustering
   - Target: Silhouette Score minimal 0.70
   - Proses:
     - Pemilihan algoritma clustering
     - Penentuan jumlah cluster optimal
     - Evaluasi dengan silhouette score
     - Iterasi untuk optimasi jika diperlukan

3. Analisis dan Interpretasi Cluster
   - Format dokumentasi per cluster:
     ```
     Cluster X:
     - Statistik deskriptif fitur-fitur utama
     - Karakteristik khusus cluster
     - Interpretasi bisnis/domain
     - Insight dan pola yang ditemukan
     ```

#### 3.3 Tahap Classification (Supervised Learning)

1. Persiapan Data
   - Penggabungan dataset dengan label hasil clustering
   - Split dataset:
     - Training set
     - Testing set

2. Pengembangan Model
   - Implementasi 2 algoritma klasifikasi berbeda
   - Target performa:
     - Accuracy ≥ 92%
     - F1-Score ≥ 92%
     - Berlaku untuk both training dan testing set

3. Evaluasi Model
   - Metrics yang diukur:
     - Accuracy
     - F1-Score
     - Precision
     - Recall
   - Perbandingan performa kedua algoritma
   - Analisis confusion matrix

### 4. DELIVERABLES

#### 4.1 File Utama
1. Clustering Notebook (.ipynb)
   - Dokumentasi proses clustering
   - Visualisasi hasil
   - Interpretasi cluster

2. Classification Notebook (.ipynb)
   - Implementasi kedua algoritma
   - Perbandingan performa
   - Analisis hasil

3. Dataset Files
   - Original dataset (.csv)
   - Dataset dengan label clustering (.csv)

#### 4.2 Dokumentasi
1. Interpretasi Cluster
   - Analisis statistik per cluster
   - Karakteristik cluster
   - Business insights

2. Performa Model
   - Metrik evaluasi clustering
   - Metrik evaluasi classification
   - Perbandingan algoritma

### 5. KRITERIA KEBERHASILAN

#### 5.1 Kriteria Clustering
1. Feature Selection
   - Dokumentasi perbandingan sebelum dan sesudah
   - Justifikasi pemilihan fitur

2. Performa Clustering
   - Silhouette Score ≥ 0.70
   - Interpretasi cluster yang komprehensif
   - Dokumentasi karakteristik tiap cluster

#### 5.2 Kriteria Classification
1. Implementasi Model
   - Minimum 2 algoritma berbeda
   - Dokumentasi proses training

2. Performa Model
   - Accuracy ≥ 92%
   - F1-Score ≥ 92%
   - Konsisten pada training dan testing set

### 6. TIPS IMPLEMENTASI

1. Clustering
   - Lakukan eksplorasi data yang mendalam sebelum clustering
   - Pertimbangkan scaling/normalisasi data
   - Eksperimen dengan berbagai jumlah cluster
   - Dokumentasikan setiap percobaan

2. Classification
   - Gunakan cross-validation
   - Implementasikan hyperparameter tuning
   - Analisis feature importance
   - Dokumentasikan proses optimasi

### 7. POTENSIAL CHALLENGES

1. Dataset
   - Menemukan dataset yang sesuai kriteria
   - Kualitas data yang mungkin tidak optimal
   - Kebutuhan preprocessing yang ekstensif

2. Performa
   - Mencapai silhouette score target (0.70)
   - Mencapai accuracy dan F1-score target (92%)
   - Konsistensi performa pada training dan testing

3. Interpretasi
   - Menghasilkan interpretasi cluster yang bermakna
   - Menjelaskan pola dan insights yang ditemukan
   - Memberikan rekomendasi berdasarkan hasil

### 8. CHECKLIST PROYEK

#### Persiapan
- [ ] Dataset teridentifikasi
- [ ] Environment setup complete
- [ ] Initial EDA completed

#### Clustering
- [ ] Data preprocessing completed
- [ ] Feature selection implemented
- [ ] Clustering model developed
- [ ] Silhouette score ≥ 0.70 achieved
- [ ] Cluster interpretation documented

#### Classification
- [ ] Dataset with labels prepared
- [ ] Two classification algorithms implemented
- [ ] Performance metrics ≥ 92% achieved
- [ ] Model comparison completed

#### Dokumentasi
- [ ] Clustering notebook completed
- [ ] Classification notebook completed
- [ ] All required CSV files generated
- [ ] Final report/documentation completed
