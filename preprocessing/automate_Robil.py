"""
automate_Robil.py
=======================================
Script otomatisasi preprocessing dataset harga emas (Gold Price Dataset).
Mengkonversi workflow eksperimen dari notebook menjadi fungsi modular
yang siap digunakan untuk pipeline machine learning.

Struktur folder yang diharapkan:
    root/
    ├── dataset/
    │   └── gold_raw.csv          ← input
    └── preprocessing/
        ├── automate_Robil.py     ← script ini
        └── gold_preprocessing/   ← output otomatis dibuat

Penggunaan:
    python preprocessing/automate_Robil.py

Output:
    preprocessing/gold_preprocessing/gold_train.csv
    preprocessing/gold_preprocessing/gold_test.csv
    preprocessing/gold_preprocessing/gold_full_preprocessed.csv
"""

import pandas as pd
import numpy as np
import os
import sys
import logging
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

# ─── Konfigurasi Logging ────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


# ─── Konstanta ──────────────────────────────────────────────────────────────
NUMERIC_COLS = ['USD (AM)', 'USD (PM)', 'GBP (AM)', 'GBP (PM)', 'EURO (AM)', 'EURO (PM)']

FEATURE_COLS = [
    'USD (AM)', 'GBP (AM)', 'GBP (PM)', 'EURO (AM)', 'EURO (PM)',
    'USD_spread', 'GBP_spread', 'EURO_spread',
    'USD_MA7', 'USD_MA30', 'USD_volatility',
    'year', 'month', 'dow'
]

SCALE_COLS = [
    'USD (AM)', 'GBP (AM)', 'GBP (PM)', 'EURO (AM)', 'EURO (PM)',
    'USD_spread', 'GBP_spread', 'EURO_spread',
    'USD_MA7', 'USD_MA30', 'USD_volatility'
]

TARGET_COL   = 'USD (PM)'
TEST_SIZE    = 0.2
RANDOM_STATE = 42


# ─── Fungsi-Fungsi Preprocessing ────────────────────────────────────────────

def load_data(filepath: str) -> pd.DataFrame:
    """
    Memuat dataset dari file CSV.

    Args:
        filepath: Path ke file CSV dataset mentah.

    Returns:
        DataFrame yang sudah dimuat.
    """
    logger.info(f"Memuat dataset dari: {filepath}")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File tidak ditemukan: {filepath}")

    df = pd.read_csv(filepath)
    logger.info(f"Dataset dimuat — Shape: {df.shape}")
    logger.info(f"Kolom: {list(df.columns)}")
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Menangani missing values menggunakan median imputation.
    (Sama dengan notebook: SimpleImputer strategy='median')
    """
    logger.info("Menangani missing values...")
    before = df.isnull().sum().sum()

    imputer = SimpleImputer(strategy='median')
    df[NUMERIC_COLS] = imputer.fit_transform(df[NUMERIC_COLS])

    after = df.isnull().sum().sum()
    logger.info(f"Missing values: {before} → {after}")
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Menghapus baris duplikat berdasarkan kolom Date.
    (Sama dengan notebook: drop_duplicates subset=['Date'])
    """
    logger.info("Menghapus data duplikat...")
    before = len(df)
    df = df.drop_duplicates(subset=['Date'])
    after = len(df)
    logger.info(f"Duplikat dihapus: {before - after} baris")
    return df


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Membuat fitur-fitur turunan dari data harga emas.
    (Sama dengan notebook cell 5.4)

    Fitur baru:
        - USD_spread, GBP_spread, EURO_spread : Selisih PM - AM
        - USD_MA7, USD_MA30                   : Moving average 7 dan 30 hari
        - USD_volatility                       : Std rolling 7 hari
        - year, month, dow                     : Fitur waktu
    """
    logger.info("Melakukan feature engineering...")

    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)

    # Spread AM-PM
    df['USD_spread']  = df['USD (PM)'] - df['USD (AM)']
    df['GBP_spread']  = df['GBP (PM)'] - df['GBP (AM)']
    df['EURO_spread'] = df['EURO (PM)'] - df['EURO (AM)']

    # Moving Average
    df['USD_MA7']  = df['USD (PM)'].rolling(window=7,  min_periods=1).mean()
    df['USD_MA30'] = df['USD (PM)'].rolling(window=30, min_periods=1).mean()

    # Volatilitas
    df['USD_volatility'] = df['USD (PM)'].rolling(window=7, min_periods=1).std().fillna(0)

    # Fitur waktu
    df['year']  = df['Date'].dt.year
    df['month'] = df['Date'].dt.month
    df['dow']   = df['Date'].dt.dayofweek  # 0=Senin, 4=Jumat

    logger.info(f"Fitur baru ditambahkan — Shape baru: {df.shape}")
    return df


def handle_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Menangani outlier pada fitur spread menggunakan clipping percentile.
    (Sama dengan notebook cell 5.5: clip 1%-99%)
    """
    logger.info("Menangani outlier pada fitur spread...")
    clip_cols = ['USD_spread', 'GBP_spread', 'EURO_spread']
    for col in clip_cols:
        q_low  = df[col].quantile(0.01)
        q_high = df[col].quantile(0.99)
        df[col] = df[col].clip(lower=q_low, upper=q_high)
        logger.info(f"  {col}: di-clip ke [{q_low:.3f}, {q_high:.3f}]")
    return df


def normalize_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalisasi fitur numerik menggunakan MinMaxScaler.
    (Sama dengan notebook cell 5.6)
    """
    logger.info("Melakukan normalisasi fitur (MinMaxScaler)...")
    scaler = MinMaxScaler()
    df[SCALE_COLS] = scaler.fit_transform(df[SCALE_COLS])
    logger.info(f"Normalisasi selesai pada {len(SCALE_COLS)} kolom.")
    return df


def split_data(df: pd.DataFrame):
    """
    Split data train-test 80:20.
    (Sama dengan notebook cell 5.7: shuffle=False untuk time-series)
    """
    logger.info(f"Splitting data — test_size={TEST_SIZE}, shuffle=False (time-series)...")
    X = df[FEATURE_COLS]
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, shuffle=False
    )
    logger.info(f"Train : {len(X_train):,} sampel")
    logger.info(f"Test  : {len(X_test):,} sampel")
    return X_train, X_test, y_train, y_test


def save_preprocessed_data(X_train, X_test, y_train, y_test, output_dir: str):
    """
    Menyimpan data yang sudah diproses ke file CSV.
    (Sama dengan notebook cell 5.8)

    Output:
        gold_train.csv
        gold_test.csv
        gold_full_preprocessed.csv
    """
    os.makedirs(output_dir, exist_ok=True)

    train_df = X_train.copy()
    train_df[TARGET_COL] = y_train.values

    test_df = X_test.copy()
    test_df[TARGET_COL] = y_test.values

    full_df = pd.concat([train_df, test_df])

    train_path = os.path.join(output_dir, 'gold_train.csv')
    test_path  = os.path.join(output_dir, 'gold_test.csv')
    full_path  = os.path.join(output_dir, 'gold_full_preprocessed.csv')

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path,   index=False)
    full_df.to_csv(full_path,   index=False)

    logger.info(f"✅ Train data disimpan   : {train_path}  {train_df.shape}")
    logger.info(f"✅ Test data disimpan    : {test_path}   {test_df.shape}")
    logger.info(f"✅ Full data disimpan    : {full_path}   {full_df.shape}")


def run_preprocessing(input_path: str, output_dir: str) -> pd.DataFrame:
    """
    Menjalankan seluruh pipeline preprocessing secara otomatis.
    Urutan tahapan sama persis dengan notebook Eksperimen_Robil.ipynb.
    """
    logger.info("=" * 60)
    logger.info("🚀 MEMULAI PIPELINE PREPROCESSING HARGA EMAS")
    logger.info("=" * 60)

    df = load_data(input_path)           # Step 1 - Load (notebook cell 3)
    df = handle_missing_values(df)       # Step 2 - Missing values (cell 5.2)
    df = remove_duplicates(df)           # Step 3 - Duplikat (cell 5.3)
    df = feature_engineering(df)         # Step 4 - Feature eng (cell 5.4)
    df = handle_outliers(df)             # Step 5 - Outlier (cell 5.5)
    df = normalize_features(df)          # Step 6 - Normalisasi (cell 5.6)

    X_train, X_test, y_train, y_test = split_data(df)          # Step 7 - Split (cell 5.7)
    save_preprocessed_data(X_train, X_test, y_train, y_test, output_dir)  # Step 8 - Save (cell 5.8)

    logger.info("=" * 60)
    logger.info("✅ PREPROCESSING SELESAI!")
    logger.info("=" * 60)

    full_df       = X_train.copy()
    full_df[TARGET_COL] = y_train.values
    full_test     = X_test.copy()
    full_test[TARGET_COL] = y_test.values
    return pd.concat([full_df, full_test])


# ─── Entry Point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Script berada di: root/preprocessing/automate_Robil.py
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))   # → root/preprocessing/
    REPO_DIR   = os.path.dirname(SCRIPT_DIR)                   # → root/

    INPUT_PATH = os.path.join(REPO_DIR,   'dataset', 'gold_raw.csv')
    OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'gold_preprocessing')

    result = run_preprocessing(
        input_path=INPUT_PATH,
        output_dir=OUTPUT_DIR
    )

    print(f"\n📊 Preview data hasil preprocessing:")
    print(result.head())
    print(f"\nShape final: {result.shape}")
