"""
Halaman: Rekomendasi Kebutuhan Bahan Baku
INI DASHBOARD UNTUK OWNER
Alur: histori -> forecast RF rekursif -> explode BOM 2 level (Menu -> Production -> Raw)
      -> integrasi Master Inventory (Hanya Konversi Unit Pack)
rekomendasi.py
"""

import numpy as np
import pandas as pd
import joblib
import os
import io
import streamlit as st
import plotly.express as px
from datetime import timedelta

# Mendapatkan path absolut dari lokasi file rekomendasi.py saat ini (folder 'modules')
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# Mundur satu folder ke atas untuk mendapatkan Root Folder (STREAMLIT-SKRIPSI)
BASE_DIR = os.path.dirname(CURRENT_DIR)

# Definisikan ulang semua rute file berdasarkan BASE_DIR
# MODEL_PATH = os.path.join(BASE_DIR, "models", "model_lr_v2.pkl") #<----- untuk test pilih model manual
MODEL_PATH = os.path.join(BASE_DIR, "models", "model_production.pkl") #<---- Otomatis menggunakan model yang di deploy dari app_dev.py
HISTORY_PATH = os.path.join(BASE_DIR, "Dataset", "df_full.csv")
BOM_FILE = os.path.join(BASE_DIR, "Dataset", "SOKO - Master Menu Soko.xlsx")

FEATURES = [
    "day_of_week", "month", "is_weekend", "week_of_month",
    "lag_1", "lag_3", "lag_7", "lag_14",
    "rolling_7", "rolling_14", "rolling_std_7",
]

# ---------------------------------------------------------------------------
# LOADERS (cached)
# ---------------------------------------------------------------------------

@st.cache_resource
def load_model(path=MODEL_PATH):
    return joblib.load(path)

@st.cache_data
def load_history(path=HISTORY_PATH):
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df["menu"] = df["menu"].str.strip().str.lower()
    return df

@st.cache_data
def load_bom(file=BOM_FILE):
    # Level 1: Menu -> Bahan
    bom_bar = pd.read_excel(file, sheet_name="Menu Bar", header=4)
    bom_kitchen = pd.read_excel(file, sheet_name="Menu Kitchen", header=4)
    bom_all = pd.concat([bom_bar, bom_kitchen], ignore_index=True)
    bom_all.columns = bom_all.columns.str.strip()
    bom_all = bom_all.dropna(how="all")
    
    # PERBAIKAN FATAL: Ubah sel kosong/spasi jadi NaN murni sebelum di-ffill
    bom_all["Menu"] = bom_all["Menu"].replace(r'^\s*$', np.nan, regex=True)
    bom_all["Menu"] = bom_all["Menu"].ffill().astype(str).str.strip().str.lower()
    
    bom_all["Bahan"] = bom_all["Bahan"].replace(r'^\s*$', np.nan, regex=True)
    bom_all = bom_all.dropna(subset=["Bahan"])
    bom_all["Bahan"] = bom_all["Bahan"].astype(str).str.strip().str.lower()
    
    bom_all["Recipe QTY"] = pd.to_numeric(bom_all["Recipe QTY"], errors="coerce").fillna(0)
    bom_all = bom_all.rename(columns={
        "Menu": "menu", "Bahan": "bahan", "Recipe QTY": "recipe_qty", "Satuan": "satuan",
    })[["menu", "bahan", "recipe_qty", "satuan"]]

    # Level 2: Production -> Raw
    prod_bar = pd.read_excel(file, sheet_name="Production Bar", header=4)
    prod_kitchen = pd.read_excel(file, sheet_name="Production Kitchen", header=4)
    prod_all = pd.concat([prod_bar, prod_kitchen], ignore_index=True)
    prod_all.columns = prod_all.columns.str.strip()
    prod_all = prod_all.dropna(how="all")
    
    # PERBAIKAN FATAL LEVEL 2
    prod_all["Menu"] = prod_all["Menu"].replace(r'^\s*$', np.nan, regex=True)
    prod_all["Menu"] = prod_all["Menu"].ffill().astype(str).str.strip().str.lower()
    
    prod_all["Bahan"] = prod_all["Bahan"].replace(r'^\s*$', np.nan, regex=True)
    prod_all["is_yield"] = prod_all["Bahan"].isna()
    
    prod_all["Bahan"] = prod_all["Bahan"].astype(str).str.strip().str.lower()
    prod_all["Recipe QTY"] = pd.to_numeric(prod_all["Recipe QTY"], errors="coerce").fillna(0)

    yield_df = prod_all[prod_all["is_yield"]].rename(
        columns={"Menu": "parent", "Recipe QTY": "yield_qty"}
    )[["parent", "yield_qty"]]

    prod_detail = prod_all[~prod_all["is_yield"]].rename(columns={
        "Menu": "parent", "Bahan": "child", "Recipe QTY": "qty_prod", "Satuan": "satuan_prod",
    })[["parent", "child", "qty_prod", "satuan_prod"]]

    prod_detail = prod_detail.merge(yield_df, on="parent", how="left")
    prod_detail["qty_per_unit"] = prod_detail["qty_prod"] / prod_detail["yield_qty"]

    return bom_all, prod_detail
# @st.cache_data
# def load_bom(file=BOM_FILE):
#     # Level 1: Menu -> Bahan
#     bom_bar = pd.read_excel(file, sheet_name="Menu Bar", header=4)
#     bom_kitchen = pd.read_excel(file, sheet_name="Menu Kitchen", header=4)
#     bom_all = pd.concat([bom_bar, bom_kitchen], ignore_index=True)
#     bom_all.columns = bom_all.columns.str.strip()
#     bom_all = bom_all.dropna(how="all")
#     bom_all["Menu"] = bom_all["Menu"].ffill().astype(str).str.strip().str.lower()
#     bom_all["Bahan"] = bom_all["Bahan"].astype(str).str.strip().str.lower()
#     bom_all = bom_all.dropna(subset=["Bahan"])
#     bom_all["Recipe QTY"] = pd.to_numeric(bom_all["Recipe QTY"], errors="coerce").fillna(0)
#     bom_all = bom_all.rename(columns={
#         "Menu": "menu", "Bahan": "bahan", "Recipe QTY": "recipe_qty", "Satuan": "satuan",
#     })[["menu", "bahan", "recipe_qty", "satuan"]]

#     # Level 2: Production -> Raw
#     prod_bar = pd.read_excel(file, sheet_name="Production Bar", header=4)
#     prod_kitchen = pd.read_excel(file, sheet_name="Production Kitchen", header=4)
#     prod_all = pd.concat([prod_bar, prod_kitchen], ignore_index=True)
#     prod_all.columns = prod_all.columns.str.strip()
#     prod_all = prod_all.dropna(how="all")
#     prod_all["Menu"] = prod_all["Menu"].ffill().astype(str).str.strip().str.lower()
#     prod_all["is_yield"] = prod_all["Bahan"].isna()
#     prod_all["Bahan"] = prod_all["Bahan"].astype(str).str.strip().str.lower()
#     prod_all["Recipe QTY"] = pd.to_numeric(prod_all["Recipe QTY"], errors="coerce").fillna(0)

#     yield_df = prod_all[prod_all["is_yield"]].rename(
#         columns={"Menu": "parent", "Recipe QTY": "yield_qty"}
#     )[["parent", "yield_qty"]]

#     prod_detail = prod_all[~prod_all["is_yield"]].rename(columns={
#         "Menu": "parent", "Bahan": "child", "Recipe QTY": "qty_prod", "Satuan": "satuan_prod",
#     })[["parent", "child", "qty_prod", "satuan_prod"]]

#     prod_detail = prod_detail.merge(yield_df, on="parent", how="left")
#     prod_detail["qty_per_unit"] = prod_detail["qty_prod"] / prod_detail["yield_qty"]

#     return bom_all, prod_detail

@st.cache_data
def load_inventory_master(file=BOM_FILE):
    """Membaca sheet Master Inventory hanya untuk konversi Unit Pack"""
    df_raw = pd.read_excel(file, sheet_name="Master", header=None)
    row_index = df_raw[df_raw.isin(['Remark']).any(axis=1)].index[0]
    new_header = df_raw.iloc[row_index]
    
    df_master = df_raw[row_index + 1:].copy()
    df_master.columns = new_header
    df_master.columns = df_master.columns.astype(str).str.replace('\n', '').str.strip()
    
    df_master = df_master.rename(columns={
        "Remark": "bahan",
        "Unit Pack": "unit_pack",
        "Qty": "qty_per_pack"
    })
    
    df_master["bahan"] = df_master["bahan"].astype(str).str.strip().str.lower()
    df_master["qty_per_pack"] = pd.to_numeric(df_master["qty_per_pack"], errors="coerce").fillna(1)
    
    return df_master[["bahan", "unit_pack", "qty_per_pack"]]


# ---------------------------------------------------------------------------
# FORECASTING & BOM EXPLOSION
# ---------------------------------------------------------------------------

def forecast_future(df_full, model, menu_list, start_date, n_days):
    all_preds = []
    for menu in menu_list:
        hist = df_full[df_full["menu"] == menu].sort_values("date")
        series = dict(zip(hist["date"], hist["qty"]))
        for i in range(n_days):
            target_date = pd.Timestamp(start_date) + timedelta(days=i)
            last_7 = [series.get(target_date - timedelta(days=d), np.nan) for d in range(1, 8)]
            last_14 = [series.get(target_date - timedelta(days=d), np.nan) for d in range(1, 15)]
            row = {
                "day_of_week": target_date.dayofweek, "month": target_date.month,
                "is_weekend": int(target_date.dayofweek >= 5), "week_of_month": (target_date.day - 1) // 7 + 1,
                "lag_1": series.get(target_date - timedelta(days=1), np.nan),
                "lag_3": series.get(target_date - timedelta(days=3), np.nan),
                "lag_7": series.get(target_date - timedelta(days=7), np.nan),
                "lag_14": series.get(target_date - timedelta(days=14), np.nan),
                "rolling_7": np.nanmean(last_7), "rolling_14": np.nanmean(last_14), "rolling_std_7": np.nanstd(last_7),
            }
            X_row = pd.DataFrame([row])[FEATURES].fillna(0)
            y_pred = max(0.0, float(model.predict(X_row)[0]))
            series[target_date] = y_pred
            all_preds.append({"date": target_date, "menu": menu, "qty_pred": round(y_pred)})
    return pd.DataFrame(all_preds)

def explode_bom(pred_df, bom_all, prod_detail, qty_col="qty_pred"):
    menu_join = pred_df.merge(bom_all, on="menu", how="left")
    full_bom = menu_join.merge(prod_detail, left_on="bahan", right_on="parent", how="left")
    full_bom["final_bahan"] = full_bom["child"].fillna(full_bom["bahan"])
    full_bom["final_qty"] = full_bom[qty_col] * full_bom["recipe_qty"] * full_bom["qty_per_unit"].fillna(1)
    full_bom["final_satuan"] = full_bom["satuan_prod"].fillna(full_bom["satuan"])
    return full_bom.groupby(["date", "final_bahan", "final_satuan"])["final_qty"].sum().reset_index().rename(
        columns={"final_bahan": "bahan", "final_satuan": "satuan", "final_qty": "kebutuhan"}
    )

@st.cache_data
def load_menu_categories(file=BOM_FILE):
    """Membaca Divisi, Kategori, dan HPP dari sheet Kesimpulan Menu"""
    df_kesimpulan = pd.read_excel(file, sheet_name="Kesimpulan Menu", header=3)
    df_kesimpulan.columns = df_kesimpulan.columns.str.strip()
    
    # Kolom B: Divisi, Kolom C: Kategori, Kolom D: Menu, Kolom G: HPP Net, Kolom H: Harga Jual
    required_cols = ['Divisi', 'Kategori', 'Menu', 'HPP Net', 'Harga Jual']
    if all(col in df_kesimpulan.columns for col in required_cols):
        cat_df = df_kesimpulan[['Divisi', 'Kategori', 'Menu', 'HPP Net', 'Harga Jual']].copy()
        cat_df = cat_df.dropna(subset=['Menu'])
        cat_df['Menu'] = cat_df['Menu'].astype(str).str.strip().str.lower()
        cat_df['Divisi'] = cat_df['Divisi'].astype(str).str.strip().str.title()
        cat_df['Kategori'] = cat_df['Kategori'].astype(str).str.strip().str.title()
        cat_df['HPP Net'] = pd.to_numeric(cat_df['HPP Net'], errors='coerce').fillna(0)
        cat_df['Harga Jual'] = pd.to_numeric(cat_df['Harga Jual'], errors='coerce').fillna(0)
        
        return cat_df.rename(columns={
            'Menu': 'menu', 
            'Divisi': 'divisi',
            'Kategori': 'kategori', 
            'HPP Net': 'hpp_net',
            'Harga Jual': 'harga_jual'
        })
    return None


# ---------------------------------------------------------------------------
# STREAMLIT PAGE (UI)
# ---------------------------------------------------------------------------

def render():
    st.title("Estimasi Kebutuhan Bahan Baku")
    meta_path = os.path.join(BASE_DIR, "models", "production_meta.json")
    if os.path.exists(meta_path):
        import json
        with open(meta_path) as f:
            meta = json.load(f)
        st.caption(f"🔧 Model aktif di produksi: **{meta['algorithm']}** · deployed {meta['deployed_at']}")
    
    st.subheader("The Soko Coffee Tea Chocolate")
    
    # =========================================================================
    # 1. LOAD DATA DI AWAL
    # =========================================================================
    df_full = load_history()
    model = load_model()
    bom_all, prod_detail = load_bom()
    df_master = load_inventory_master()
    df_cat = load_menu_categories()

    menu_list = df_full["menu"].unique()

    # =========================================================================
    # 2. LOGIKA FILTER MENU RACIKAN (Membuang RTD Bar, Menyelamatkan Kitchen)
    # =========================================================================
    # Hitung jumlah bahan baku per menu
    ingredient_counts = bom_all.groupby('menu')['bahan'].nunique().reset_index()
    ingredient_counts.columns = ['menu', 'ing_count']
    
    if df_cat is not None:
        menu_info = ingredient_counts.merge(df_cat[['menu', 'divisi']], on='menu', how='left')
        # ATURAN EMAS: Menu diakui valid JIKA dia masuk divisi 'Kitchen' ATAU jumlah bahannya > 1
        valid_menus_df = menu_info[
            (menu_info['divisi'].str.lower() == 'kitchen') | 
            (menu_info['ing_count'] > 1) | 
            (menu_info['divisi'].isna())
        ]
        valid_menus_bom = valid_menus_df['menu'].tolist()
    else:
        valid_menus_bom = ingredient_counts[ingredient_counts['ing_count'] > 1]['menu'].tolist()

    # =========================================================================
    # 3. RINGKASAN PENJUALAN 7 HARI TERAKHIR (Tanpa Air Mineral)
    # =========================================================================
    last_date_in_data = df_full["date"].max()
    default_start_date = last_date_in_data + pd.Timedelta(days=1)
    
    tgl_awal = last_date_in_data - pd.Timedelta(days=6)
    data_minggu_lalu = df_full[(df_full['date'] >= tgl_awal) & (df_full['date'] <= last_date_in_data)]
    
    # Total porsi dibiarkan keseluruhan (karena air mineral tetap menghasilkan uang)
    total_porsi = data_minggu_lalu['qty'].sum() 
    
    # Filter pencarian menu terlaris khusus untuk menu racikan yang valid
    data_minggu_lalu_racikan = data_minggu_lalu[data_minggu_lalu['menu'].isin(valid_menus_bom)]
    if not data_minggu_lalu_racikan.empty:
        top_menu = data_minggu_lalu_racikan.groupby('menu')['qty'].sum().idxmax()
        top_menu_qty = data_minggu_lalu_racikan.groupby('menu')['qty'].sum().max()
    else:
        top_menu = "-"
        top_menu_qty = 0

    st.markdown("### Ringkasan Performa Transaksi 7 Hari Terakhir")
    m1, m2, m3 = st.columns(3)
    with m1:
        with st.container(border=True):
            st.metric("Total Porsi Terjual (Keseluruhan)", f"{int(total_porsi):,} Porsi")
    with m2:
        with st.container(border=True):
            st.metric("Menu Paling Laku", top_menu.title())
    with m3:
        with st.container(border=True):
            st.metric("Volume Menu Racikan Terjual", f"{int(top_menu_qty):,} Porsi")
            
    st.divider()

    # =========================================================================
    # 4. PESAN TEMPAT PROYEKSI & UI KALENDER
    # =========================================================================
    proyeksi_container = st.container()

    col1, col2 = st.columns([2, 1])
    with col1:
        start_date = st.date_input("Prediksi mulai tanggal (Otomatis H+1)", value=default_start_date.date())
    with col2:
        st.info("⏱️ Mode Analisis: Akumulasi Mingguan (7 Hari)")
        n_days = 7

    # =========================================================================
    # 5. PROSES PREDIKSI, KEUANGAN, & MAPPING DIVISI BAHAN BAKU
    # =========================================================================
    with st.spinner("Menghitung prediksi & kebutuhan bahan baku..."):
        pred_future = forecast_future(df_full, model, menu_list, start_date, n_days)
        
        df_cat = load_menu_categories()
        if df_cat is not None:
            pred_with_cat = pred_future.merge(df_cat, on="menu", how="left")
            pred_with_cat['kategori'] = pred_with_cat['kategori'].fillna("Lainnya")
            pred_with_cat['divisi'] = pred_with_cat['divisi'].fillna("Lainnya")
            
            pred_with_cat['estimasi_omset'] = pred_with_cat['qty_pred'] * pred_with_cat['harga_jual']
            pred_with_cat['estimasi_hpp'] = pred_with_cat['qty_pred'] * pred_with_cat['hpp_net']
            
            total_omset = pred_with_cat['estimasi_omset'].sum()
            total_hpp_modal = pred_with_cat['estimasi_hpp'].sum()
            estimated_profit = total_omset - total_hpp_modal
            total_porsi_pred = pred_with_cat['qty_pred'].sum()
            
            with proyeksi_container:
                st.markdown("### 💰 Proyeksi Keuangan & Bisnis Minggu Depan")
                k1, k2, k3, k4 = st.columns(4)
                with k1:
                    with st.container(border=True):
                        st.metric("Total Porsi Prediksi", f"{int(total_porsi_pred):,} Porsi")
                with k2:
                    with st.container(border=True):
                        st.metric("Proyeksi Omset", f"Rp {total_omset:,.0f}")
                with k3:
                    with st.container(border=True):
                        st.metric("Estimasi Anggaran Belanja (RAB)", f"Rp {total_hpp_modal:,.0f}")
                with k4:
                    with st.container(border=True):
                        st.metric("Estimasi Laba Kotor", f"Rp {estimated_profit:,.0f}")
                st.divider()
        else:
            pred_with_cat = pred_future

        # =====================================================================
        # 6. EKSPLOSI BOM & MAPPING KATEGORI/DIVISI BAHAN BAKU
        # =====================================================================
        raw_needs = explode_bom(pred_future, bom_all, prod_detail, qty_col="qty_pred")
        rekom = raw_needs.groupby("bahan", as_index=False).agg(satuan=("satuan", "first"), kebutuhan=("kebutuhan", "sum"))
        rekom = rekom.merge(df_master.drop_duplicates(subset='bahan'), on="bahan", how="left")
        
        # --- LOGIKA OTOMATIS MAPPING DIVISI BAHAN BAKU (BAR / KITCHEN) ---
        if df_cat is not None:
            # Lacak bahan langsung dari Menu Bar/Kitchen
            mat_div_1 = bom_all.merge(df_cat[['menu', 'divisi']], on='menu', how='left')[['bahan', 'divisi']]
            # Lacak bahan dari Production Bar/Kitchen
            prod_to_menu = bom_all.rename(columns={'bahan': 'parent', 'menu': 'menu'})
            mat_div_2 = prod_detail.merge(prod_to_menu[['parent', 'menu']], on='parent', how='left')
            mat_div_2 = mat_div_2.merge(df_cat[['menu', 'divisi']], on='menu', how='left')[['child', 'divisi']].rename(columns={'child': 'bahan'})
            
            material_division = pd.concat([mat_div_1, mat_div_2]).dropna(subset=['divisi'])
            material_division = material_division.drop_duplicates().groupby('bahan')['divisi'].agg(lambda x: ' / '.join(x.unique())).reset_index()
            
            rekom = rekom.merge(material_division, on="bahan", how="left")
            rekom['divisi'] = rekom['divisi'].fillna("Umum / Lainnya")
        else:
            rekom['divisi'] = "Umum"

        rekom["jumlah_beli"] = np.ceil(rekom["kebutuhan"] / rekom["qty_per_pack"].fillna(1)).astype(int)
        rekom["unit_pack"] = rekom["unit_pack"].fillna(rekom["satuan"]) 
        rekom = rekom.loc[:, ~rekom.columns.duplicated()]

    # =========================================================================
    # 7. TABEL REKOMENDASI DENGAN FILTER KATEGORI/DIVISI
    # =========================================================================
    end_date = start_date + pd.Timedelta(days=n_days - 1)
    st.info(f"📅 Menampilkan hasil peramalan untuk periode: **{start_date.strftime('%d %b %Y')}** sampai **{end_date.strftime('%d %b %Y')}**")

    rekom = rekom.sort_values("kebutuhan", ascending=False)
    st.subheader("Tabel Estimasi Kebutuhan Bahan Baku")
    
    rekom_display = rekom.rename(columns={
        "bahan": "Bahan Baku", "kebutuhan": "Kebutuhan Netto", 
        "satuan": "Satuan Dasar", "jumlah_beli": "Estimasi Kebutuhan", 
        "unit_pack": "Jenis Kemasan", "divisi": "Divisi"
    })

    kolom_ui = ["Bahan Baku", "Divisi", "Kebutuhan Netto", "Satuan Dasar", "Estimasi Kebutuhan", "Jenis Kemasan"]

    # --- FILTER UI (PILIH DIVISI & PENCARIAN TEKS) ---
    f_col1, f_col2 = st.columns([1, 2])
    with f_col1:
        divisi_list = ["Semua Divisi"] + sorted(rekom_display['Divisi'].unique().tolist())
        selected_divisi = st.selectbox("Filter Berdasarkan Divisi:", divisi_list)
    with f_col2:
        search = st.text_input("Cari nama bahan baku...")

    filtered = rekom_display[kolom_ui].copy()
    
    # Eksekusi Filter Divisi
    if selected_divisi != "Semua Divisi":
        filtered = filtered[filtered["Divisi"] == selected_divisi]
        
    # Eksekusi Filter Search Text
    if search:
        filtered = filtered[filtered["Bahan Baku"].str.contains(search, case=False, na=False)] 

    st.dataframe(filtered.style.format({"Kebutuhan Netto": "{:,.1f}"}), use_container_width=True, hide_index=True)
    st.caption(
        "Kebutuhan Netto dihitung dari prediksi resep. Kolom 'Estimasi Kebutuhan' mengonversi total netto "
        "menjadi jumlah kemasan utuh (pembulatan ke atas) berdasarkan master data inventory. "
        "Silakan bandingkan angka Estimasi Kebutuhan ini dengan sisa fisik di gudang sebelum berbelanja."
    )
    # =========================================================================
    # 8. VISUALISASI TOP 5 (BAR VS KITCHEN) 
    # =========================================================================
    st.markdown("### Analisis Tren Menu Berdasarkan Divisi")
    st.caption(
        "💡 **Petunjuk Operasional:** Grafik di bawah menampilkan proyeksi 5 menu racikan terlaris untuk masing-masing divisi (Bar & Kitchen) minggu depan. "
        "Informasi ini berfungsi membantu Head Barista dan Head Chef memprioritaskan persiapan bahan baku (*pre-prep*) serta memastikan ketersediaan stok menu favorit pelanggan."
    )
    # Filter menggunakan valid_menus_bom yang sudah ditetapkan di atas
    pred_menu_clean = pred_future[pred_future['menu'].isin(valid_menus_bom)]
    
    if df_cat is not None:
        pred_menu_clean = pred_menu_clean.merge(df_cat[['menu', 'divisi']], on='menu', how='left')
        col_bar, col_kitchen = st.columns(2)
        
        with col_bar:
            st.markdown("**☕ Top 5 Menu Bar Paling Laku**")
            bar_data = pred_menu_clean[pred_menu_clean['divisi'].str.lower() == 'bar']
            # FILTER 0 PORSI AGAR GRAFIK TIDAK KOSONG/ANEH
            bar_data = bar_data[bar_data['qty_pred'] > 0] 
            top_bar = bar_data.groupby('menu')['qty_pred'].sum().reset_index().nlargest(5, 'qty_pred')
            
            if not top_bar.empty:
                fig_bar_menu = px.bar(top_bar, x="qty_pred", y="menu", orientation='h', labels={"qty_pred": "Porsi", "menu": "Menu Bar"}, color="qty_pred", color_continuous_scale="Teal")
                fig_bar_menu.update_layout(yaxis=dict(autorange="reversed"), margin=dict(l=10, r=10, t=10, b=10), coloraxis_showscale=False)
                st.plotly_chart(fig_bar_menu, use_container_width=True)
            else:
                st.info("Prediksi penjualan menu Bar minggu depan adalah 0.")
                
        with col_kitchen:
            st.markdown("**🍳 Top 5 Menu Kitchen Paling Laku**")
            kitchen_data = pred_menu_clean[pred_menu_clean['divisi'].str.lower() == 'kitchen']
            # FILTER 0 PORSI AGAR GRAFIK TIDAK KOSONG/ANEH
            kitchen_data = kitchen_data[kitchen_data['qty_pred'] > 0] 
            top_kitchen = kitchen_data.groupby('menu')['qty_pred'].sum().reset_index().nlargest(5, 'qty_pred')
            
            if not top_kitchen.empty:
                fig_kit_menu = px.bar(top_kitchen, x="qty_pred", y="menu", orientation='h', labels={"qty_pred": "Porsi", "menu": "Menu Kitchen"}, color="qty_pred", color_continuous_scale="Oranges")
                fig_kit_menu.update_layout(yaxis=dict(autorange="reversed"), margin=dict(l=10, r=10, t=10, b=10), coloraxis_showscale=False)
                st.plotly_chart(fig_kit_menu, use_container_width=True)
            else:
                st.info("Prediksi menu Kitchen kosong atau tidak ada penjualan minggu depan.")

    # =========================================================================
    # 8. ANALISIS POLA PENJUALAN HARIAN (KOREKSI AGREGASI HARIAN)
    # =========================================================================
    st.divider()
    st.markdown("### 📅 Pola Rata-Rata Penjualan Harian (Senin - Minggu)")
    st.caption(
        "💡 **Insight Musiman (Seasonality):** Grafik ini memperlihatkan fluktuasi rata-rata total porsi terjual per hari. "
        "Digunakan pengelola untuk mengantisipasi lonjakan akhir pekan (*weekend rush*) dalam penyiapan stok bahan baku."
    )
    
    # 1. Agregasi total porsi per tanggal (gabungkan semua menu per hari)
    total_harian = df_full.groupby('date')['qty'].sum().reset_index()
    total_harian['day_of_week'] = total_harian['date'].dt.dayofweek

    # 2. Hitung rata-rata total porsi harian berdasarkan hari
    pola_mingguan = total_harian.groupby('day_of_week')['qty'].mean().reset_index()
    hari_map = {0: 'Senin', 1: 'Selasa', 2: 'Rabu', 3: 'Kamis', 4: 'Jumat', 5: 'Sabtu', 6: 'Minggu'}
    pola_mingguan['nama_hari'] = pola_mingguan['day_of_week'].map(hari_map)
    pola_mingguan = pola_mingguan.sort_values('day_of_week')

    # 3. Visualisasi Bar Chart
    fig_pola = px.bar(
        pola_mingguan, 
        x="nama_hari", 
        y="qty",
        text_auto=".0f",
        labels={"nama_hari": "Hari", "qty": "Rata-rata Total Porsi"},
        color="qty", 
        color_continuous_scale="Viridis"
    )
    fig_pola.update_traces(textposition='outside')
    fig_pola.update_layout(
        xaxis_title="", 
        yaxis_title="Rata-rata Porsi/Hari",
        margin=dict(l=10, r=10, t=30, b=10),
        coloraxis_showscale=False
    )
    
    st.plotly_chart(fig_pola, use_container_width=True)
#^^^^ NEW

if __name__ == "__main__":
    render()