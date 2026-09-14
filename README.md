# 📈 Demand Forecasting - The Soko Coffee Tea Chocolate

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Status](https://img.shields.io/badge/Status-Completed-success.svg)]()

Repositori ini berisi *source code* untuk program Tugas Akhir (Skripsi) dengan judul **"Peramalan Kebutuhan Stok Bahan Baku Menggunakan Linear Regression dan Random Forest pada The Soko Coffee Tea Chocolate"**.

## 👤 Informasi Penulis
*   **Nama:** Aqbil Gradiansyah
*   **NIM:** 10522138
*   **Program Studi:** Sistem Informasi
*   **Universitas:** Universitas Komputer Indonesia (UNIKOM)

##  Deskripsi Proyek
Sistem ini dibangun untuk membantu **The Soko Coffee Tea Chocolate** memprediksi kebutuhan stok bahan baku harian dan bulanan guna menghindari kehabisan stok (*stockout*) atau penumpukan bahan (*overstock*). 

Proyek ini menggunakan data penjualan historis yang diekstraksi menjadi kebutuhan bahan baku mentah melalui metode **Bill of Materials (BOM) Explosion**. Data tersebut kemudian dimodelkan menggunakan dua algoritma *Machine Learning* untuk dibandingkan performanya:
1.  **Linear Regression:** Digunakan sebagai *baseline* model untuk melihat tren linear dari pergerakan data.
2.  **Random Forest:** Digunakan untuk menangkap pola data yang lebih kompleks dan non-linear.

##  Fitur Utama
*   **Data Preprocessing:** Pembersihan data penjualan dan konversi menu ke bahan baku (*BOM Explosion*).
*   **Forecasting Engine:** Prediksi kebutuhan bahan baku menggunakan Linear Regression dan Random Forest.
*   **Model Evaluation:** Perbandingan metrik akurasi (seperti RMSE, MAE, atau MAPE) antara kedua model.
*   **Owner Dashboard (`app_owner.py`):** Antarmuka interaktif bagi pemilik bisnis untuk melihat hasil *forecasting* dan mengambil keputusan pengadaan barang.

##  Teknologi yang Digunakan
*   **Bahasa Pemrograman:** Python
*   **Library Data & ML:** Pandas, NumPy, Scikit-Learn
*   **Deployment/Dashboard:** Streamlit / Tkinter (via `app_owner.py`)

##  Cara Menjalankan Program

1.  **Clone repository ini:**
    ```bash
    git clone [https://github.com/agradiansyah/demand_forecasting_skripsi_10522138.git](https://github.com/agradiansyah/demand_forecasting_skripsi_10522138.git)
    cd demand_forecasting_skripsi_10522138
    ```

2.  **Install dependencies yang dibutuhkan:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Catatan: Pastikan Anda sudah membuat file requirements.txt jika menggunakan virtual environment)*

3.  **Jalankan aplikasi utama:**
    ```bash
    python app_owner.py
    ```
    *(Gunakan `streamlit run app_owner.py` jika aplikasi dibangun menggunakan Streamlit)*

## Kesimpulan Analisis
Hasil evaluasi dari sistem ini menunjukkan perbandingan performa yang objektif antara model Linear Regression dan Random Forest, sehingga dapat merekomendasikan algoritma terbaik untuk diimplementasikan secara permanen pada manajemen inventori The Soko Coffee Tea Chocolate.

---
*© 2026 Aqbil Gradiansyah. All rights reserved.*
