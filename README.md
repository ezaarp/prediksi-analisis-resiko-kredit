# 📊 Analisis Risiko Kredit - Data Mining Project

## 🎯 Deskripsi Proyek

Proyek Tugas Besar Mata Kuliah Data Mining ini bertujuan menganalisis dataset risiko kredit untuk mengeksplorasi faktor-faktor yang memengaruhi kemungkinan gagal bayar pinjaman oleh individu. Dengan menggunakan teknik machine learning dan clustering, proyek ini menghasilkan model prediksi yang dapat membantu institusi keuangan dalam pengambilan keputusan terkait pemberian kredit.

## 👥 Tim Pengembang

**Kelompok 1 - SI4701**

| Nama | NIM |
|------|-----|
| Andrarieza Rizqi Pradana | 102022330319 |
| Muhamad Habibi Budiman | 102022300226 |
| Kias Huma Di Stepa | 102022300446 |

## 🚀 Demo Aplikasi

Aplikasi web interaktif telah dibuat menggunakan Streamlit untuk memudahkan eksplorasi data dan prediksi risiko kredit.

### Fitur Utama:
- 📈 **Dashboard Analisis**: Visualisasi data dan statistik deskriptif
- 🤖 **Prediksi Risiko**: Input data nasabah untuk prediksi status pinjaman
- 📊 **Evaluasi Model**: Metrik performa model (Accuracy: 72%, ROC AUC: 81%)
- 🎯 **Clustering Analysis**: Segmentasi nasabah berdasarkan profil risiko

## 📋 Dataset

Dataset berisi **32,581 records** dengan 12 fitur utama:

| Fitur | Deskripsi |
|-------|-----------|
| `person_age` | Usia individu |
| `person_income` | Pendapatan tahunan |
| `person_home_ownership` | Status kepemilikan rumah (RENT/OWN/MORTGAGE/OTHER) |
| `person_emp_length` | Lama bekerja (tahun) |
| `loan_intent` | Tujuan pinjaman (PERSONAL/EDUCATION/MEDICAL/VENTURE/HOMEIMPROVEMENT/DEBTCONSOLIDATION) |
| `loan_grade` | Grade pinjaman (A-G) |
| `loan_amnt` | Jumlah pinjaman |
| `loan_int_rate` | Suku bunga pinjaman |
| `loan_status` | Status pinjaman (0=Lunas, 1=Gagal Bayar) |
| `loan_percent_income` | Persentase pinjaman terhadap pendapatan |
| `cb_person_default_on_file` | Riwayat gagal bayar (Y/N) |
| `cb_person_cred_hist_length` | Panjang riwayat kredit (tahun) |

## 🛠️ Metodologi

### 1. Data Preprocessing
- **Missing Value Handling**: Imputasi median untuk numerik, mean untuk loan_int_rate
- **Outlier Treatment**: IQR method untuk menangani outlier
- **Feature Encoding**: Label Encoding untuk variabel kategorikal
- **Data Scaling**: StandardScaler untuk normalisasi fitur

### 2. Clustering Analysis
- **Dimensionality Reduction**: PCA (2 komponen utama)
- **Algorithm**: K-Means Clustering
- **Optimal Clusters**: 3 cluster (berdasarkan Silhouette Score)
- **Hyperparameter Tuning**: GridSearchCV untuk optimasi parameter

### 3. Classification Model
- **Algorithm**: Logistic Regression
- **Data Balancing**: SMOTE untuk mengatasi class imbalance
- **Feature Engineering**: Menambahkan cluster sebagai fitur tambahan
- **Hyperparameter Tuning**: GridSearchCV dengan ROC AUC scoring

## 📊 Hasil Model

### Performa Model:
- **Accuracy**: 72%
- **Precision**: 43% (untuk kelas gagal bayar)
- **Recall**: 76% (untuk kelas gagal bayar)
- **F1-Score**: 54% (untuk kelas gagal bayar)
- **ROC AUC**: 81%

### Clustering Results:
- **Silhouette Score**: 0.492
- **3 Cluster Optimal**: Berdasarkan profil risiko nasabah
- **Best Parameters**: n_clusters=3, init='k-means++', max_iter=100

## 📈 Insights Bisnis

### Faktor Risiko Utama:
1. **Riwayat Gagal Bayar**: Nasabah dengan riwayat gagal bayar memiliki risiko 3x lebih tinggi
2. **Loan-to-Income Ratio**: Rasio pinjaman terhadap pendapatan >30% meningkatkan risiko
3. **Grade Pinjaman**: Grade D-G memiliki tingkat gagal bayar lebih tinggi
4. **Tujuan Pinjaman**: Pinjaman untuk venture dan debt consolidation berisiko tinggi

### Rekomendasi:
- Implementasi scoring model untuk otomasi keputusan kredit
- Enhanced due diligence untuk nasabah dengan profil risiko tinggi
- Penyesuaian suku bunga berdasarkan cluster risiko
- Monitoring berkala untuk early warning system

## 🔬 Library dan tools yang Digunakan

- **Python 3.8+**: Bahasa pemrograman utama
- **Pandas & NumPy**: Data manipulation dan numerical computing
- **Scikit-learn**: Machine learning algorithms
- **Matplotlib & Seaborn**: Data visualization
- **Streamlit**: Web application framework
- **Jupyter Notebook**: Interactive development environment

<div align="center">
  <p><strong>⭐ Semoga Bermanfaat! ⭐</strong></p>
  <p>Made by Kelompok 1 - SI4701</p>
</div> 