#app_owner.py
import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import json
import shutil
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score

# Impor modul dasbor analitik
from modules import rekomendasi

# ---------------------------------------------------------------------------
# KONFIGURASI PATH & VARIABEL
# ---------------------------------------------------------------------------
st.set_page_config(page_title="SPK Pengadaan - The Soko", layout="wide", page_icon="☕")

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PRODUCTION_PATH = os.path.join(CURRENT_DIR, "models", "model_production.pkl")
PRODUCTION_META_PATH = os.path.join(CURRENT_DIR, "models", "production_meta.json")
HISTORY_PATH = os.path.join(CURRENT_DIR, "Dataset", "df_full.csv")

FEATURES = [
    "day_of_week", "month", "is_weekend", "week_of_month",
    "lag_1", "lag_3", "lag_7", "lag_14",
    "rolling_7", "rolling_14", "rolling_std_7",
]
TARGET = "qty"

# ---------------------------------------------------------------------------
# FUNGSI AUTO-PREPROCESSING (Di Belakang Layar)
# ---------------------------------------------------------------------------
def auto_preprocess(raw_df):
    """Otomatisasi pembersihan, agregasi, gap filling, dan feature engineering."""
    df = raw_df.copy()
    
    # 1. Hardcode Kolom sesuai format ESB agar owner tidak perlu repot memilih
    date_col, menu_col, qty_col, type_col = 'Sales Date', 'Menu Name', 'Qty', 'Type'
    
    # 2. Cleaning & Agregasi
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df[menu_col] = df[menu_col].astype(str).str.strip().str.lower()
    df[qty_col] = pd.to_numeric(df[qty_col], errors='coerce').fillna(0)
    df = df.dropna(subset=[date_col, menu_col])
    
    daily_new = df.groupby([date_col, menu_col])[qty_col].sum().reset_index()
    daily_new.rename(columns={date_col: 'date', menu_col: 'menu', qty_col: 'qty'}, inplace=True)
    
    # 3. Append Histori
    if os.path.exists(HISTORY_PATH):
        hist_df = pd.read_csv(HISTORY_PATH)
        hist_df['date'] = pd.to_datetime(hist_df['date'])
        hist_base = hist_df[['date', 'menu', 'qty']].copy()
        combined_daily = pd.concat([hist_base, daily_new], ignore_index=True)
        combined_daily = combined_daily.drop_duplicates(subset=['date', 'menu'], keep='last')
    else:
        combined_daily = daily_new

    # 4. Gap Filling
    all_dates = pd.date_range(start=combined_daily['date'].min(), end=combined_daily['date'].max(), freq='D')
    menus = combined_daily['menu'].unique()
    full_idx = pd.MultiIndex.from_product([all_dates, menus], names=['date', 'menu'])
    full_df = pd.DataFrame(index=full_idx).reset_index()
    
    merged_df = pd.merge(full_df, combined_daily, on=['date', 'menu'], how='left')
    merged_df['qty'] = merged_df['qty'].fillna(0)
    
    # 5. Feature Engineering
    merged_df = merged_df.sort_values(['menu', 'date']).reset_index(drop=True)
    merged_df['day_of_week'] = merged_df['date'].dt.dayofweek
    merged_df['month'] = merged_df['date'].dt.month
    merged_df['is_weekend'] = merged_df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
    merged_df['week_of_month'] = (merged_df['date'].dt.day - 1) // 7 + 1
    
    g = merged_df.groupby('menu')['qty']
    merged_df['lag_1'] = g.shift(1)
    merged_df['lag_3'] = g.shift(3)
    merged_df['lag_7'] = g.shift(7)
    merged_df['lag_14'] = g.shift(14)
    merged_df['rolling_7'] = merged_df.groupby('menu')['qty'].transform(lambda s: s.shift(1).rolling(7).mean())
    merged_df['rolling_14'] = merged_df.groupby('menu')['qty'].transform(lambda s: s.shift(1).rolling(14).mean())
    merged_df['rolling_std_7'] = merged_df.groupby('menu')['qty'].transform(lambda s: s.shift(1).rolling(7).std())

    final_df = merged_df.dropna().reset_index(drop=True)
    return final_df

# ---------------------------------------------------------------------------
# NAVIGASI SISTEM (SIDEBAR)
# ---------------------------------------------------------------------------
st.sidebar.title("Menu")
st.sidebar.markdown("---")
menu = st.sidebar.radio(
    "Pilih Halaman:",
    ["1. Dasbor Analitik & Pengadaan", "2. Pembaruan Data Sistem"]
)

# ===========================================================================
# HALAMAN 1: DASBOR ANALITIK & PENGADAAN (Memanggil rekomendasi.py)
# ===========================================================================
if menu == "1. Dasbor Analitik & Pengadaan":
    # Langsung panggil fungsi render dari file rekomendasi.py
    rekomendasi.render()

# ===========================================================================
# HALAMAN 2: PEMBARUAN DATA SISTEM (ONE-CLICK MAGIC + PREVIEW DATASET)
# ===========================================================================
elif menu == "2. Pembaruan Data Sistem":
    st.title("Unggah Data Penjualan Terbaru")
    st.markdown("""
    Harap unggah data penjualan 1x setiap minggu.
    Halaman ini berfungsi untuk memperbarui data transaksi terbaru dari sistem POS (ESB). 
    Cukup unggah file Excel/CSV, dan sistem akan secara otomatis membersihkan data, melatih ulang kecerdasan buatan (*machine learning*), dan menerapkan algoritma terbaik untuk memprediksi kebutuhan minggu depan.
    """)
    
    st.info("💡 **Petunjuk:** Pastikan file yang diunggah adalah hasil *export* mentah dari ESB tanpa mengubah nama kolom (Sales Date, Menu Name, Qty).")
    
    uploaded_file = st.file_uploader("Upload Data Penjualan Terbaru (CSV/Excel)", type=["csv", "xlsx"])
    skip_rows = st.number_input("Baris awal tabel header pada file ESB (Bawaan: 13):", min_value=1, value=13, step=1)
    actual_skip = int(skip_rows - 1)
    
    if uploaded_file is not None:
        try:
            # 1. BACA UNTUK PREVIEW
            if uploaded_file.name.endswith('.csv'):
                raw_df = pd.read_csv(uploaded_file, skiprows=actual_skip, encoding='utf-8-sig', on_bad_lines='skip')
            else:
                raw_df = pd.read_excel(uploaded_file, skiprows=actual_skip)
            
            # --- TAMBAHAN: TAMPILAN PREVIEW DATASET ---
            st.divider()
            st.subheader("📋 Preview Dataset Terunggah")
            
            # Metrik Ringkasan Data
            c1, c2, c3 = st.columns(3)
            c1.metric("Total Baris Data", f"{len(raw_df):,} Baris")
            c2.metric("Total Kolom", f"{len(raw_df.columns)} Kolom")
            
            # Cek Tanggal jika kolom Sales Date ada
            if 'Sales Date' in raw_df.columns:
                dates_parsed = pd.to_datetime(raw_df['Sales Date'], errors='coerce').dropna()
                if not dates_parsed.empty:
                    min_date = dates_parsed.min().strftime('%d %b %Y')
                    max_date = dates_parsed.max().strftime('%d %b %Y')
                    c3.metric("Rentang Transaksi", f"{min_date} - {max_date}")
            
            # Tampilkan 5 Baris Pertama Data Mentah
            st.dataframe(raw_df.head(5), use_container_width=True)
            
            # Expander untuk melihat struktur nama kolom
            with st.expander("Lihat Semua Nama Kolom Terbaca"):
                st.write(list(raw_df.columns))
            
            st.divider()
            # ------------------------------------------

            # TOMBOL EKSEKUSI TRAINING
            if st.button("🚀 Proses & Perbarui Rekomendasi", type="primary", use_container_width=True):
                with st.spinner("Sistem sedang memproses data dan melatih ulang algoritma. Mohon tunggu beberapa saat..."):
                    # 2. Auto-Preprocess
                    final_df = auto_preprocess(raw_df)
                    
                    # 3. Persiapan Training (Chronological Split 80:20)
                    df_sorted = final_df.sort_values('date').reset_index(drop=True)
                    X = df_sorted[FEATURES]
                    y = df_sorted[TARGET]
                    split_idx = int(len(X) * 0.8)
                    
                    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
                    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
                    
                    # 4. Auto-Train Linear Regression
                    lr = LinearRegression(fit_intercept=True)
                    lr.fit(X_train, y_train)
                    lr_rmse = root_mean_squared_error(y_test, lr.predict(X_test))
                    
                    # 5. Auto-Train Random Forest (Parameter hasil skripsi)
                    rf = RandomForestRegressor(n_estimators=200, max_depth=10, min_samples_split=5, min_samples_leaf=2, random_state=42)
                    rf.fit(X_train, y_train)
                    rf_rmse = root_mean_squared_error(y_test, rf.predict(X_test))
                    
                    # 6. Kompetisi Model & Auto-Deploy
                    if lr_rmse <= rf_rmse:
                        best_model = lr
                        algo_name = "Linear Regression"
                    else:
                        best_model = rf
                        algo_name = "Random Forest"
                        
                    # 7. Simpan Model & Data Master
                    joblib.dump(best_model, MODEL_PRODUCTION_PATH)
                    final_df.to_csv(HISTORY_PATH, index=False)
                    
                    # Simpan metadata
                    meta = {
                        "algorithm": algo_name,
                        "deployed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "rf_rmse": rf_rmse,
                        "lr_rmse": lr_rmse
                    }
                    with open(PRODUCTION_META_PATH, "w") as f:
                        json.dump(meta, f, indent=2)
                        
                    st.success("✅ **Data berhasil diperbarui!** Sistem telah menyesuaikan algoritma kecerdasan buatan dengan tren pasar terbaru.")
                    st.balloons()
                    
                    # 8. Expander khusus pembuktian dosen penguji (Sembunyi dari pandangan owner)
                    with st.expander("🔍 Lihat Detail Teknis Model (Khusus Penguji / Developer)"):
                        st.write(f"Model produksi saat ini menggunakan: **{algo_name}** (karena memiliki RMSE terendah).")
                        st.write(f"- RMSE Linear Regression : **{lr_rmse:.4f}**")
                        st.write(f"- RMSE Random Forest     : **{rf_rmse:.4f}**")
                        st.write("Data master (`df_full.csv`) telah diperbarui untuk peramalan minggu depan.")
                        
        except Exception as e:
            st.error(f"Terjadi kesalahan saat membaca file: {e}. Pastikan angka 'Baris awal tabel header' sudah pas.")