"""
preprocessing.py
----------------
King County House Sales veri seti için veri ön işleme modülü.

Veri Seti: https://www.kaggle.com/datasets/harlfoxem/housesalesprediction
Boyut: 21,613 satır, 21 sütun

Adımlar:
  1. Ham veriyi yükle
  2. Gereksiz sütunları kaldır (id, date→yıl/ay çıkar, zipcode)
  3. Outlier temizleme (bedrooms > 10)
  4. Feature engineering (bina yaşı, renovasyon, alan oranları)
  5. Log transform — fiyat dağılımı sağa çarpık (skew=4.02)
  6. StandardScaler ile ölçeklendirme
  7. Train / Validation / Test bölme (%70 / %15 / %15)
  8. İşlenmiş dosyaları kaydet

Akademik Referans (Log Transform):
    Malpezzi, S. (2003). "Hedonic Pricing Models: A Selective and
    Applied Review." In Housing Economics and Public Policy, pp. 67–89.
    → Gayrimenkul fiyat tahmininde log-doğrusal modeller standarttır.
    → Sağa çarpık fiyat dağılımını normalize eder, büyük fiyatların
      etkisini dengeler.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# ─── Dosya yolları ─────────────────────────────────────────────────────────────
RAW_PATH      = os.path.join("data", "raw", "kc_house_data.csv")
PROCESSED_DIR = os.path.join("data", "processed")
SCALER_PATH   = os.path.join("models", "scaler.pkl")
Y_SCALER_PATH = os.path.join("models", "y_scaler.pkl")

os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs("models", exist_ok=True)

TARGET_COL = "price"


# ─── 1. Veriyi Yükle ───────────────────────────────────────────────────────────
def load_data(path: str = RAW_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"[✓] Veri yüklendi: {df.shape[0]:,} satır, {df.shape[1]} sütun")
    return df


# ─── 2. Temizleme ve Sütun Düzenleme ──────────────────────────────────────────
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Tarih sütunundan satış yılı ve ayı çıkar, sonra düşür
    df["sale_year"]  = df["date"].str[:4].astype(int)
    df["sale_month"] = df["date"].str[4:6].astype(int)
    df = df.drop(columns=["id", "date"])

    # Zipcode: lat/long zaten konum bilgisini daha iyi taşıyor
    df = df.drop(columns=["zipcode"])

    # Outlier: 33 odalı ev veri giriş hatası, 10'da kes
    outlier_count = (df["bedrooms"] > 10).sum()
    df["bedrooms"] = df["bedrooms"].clip(upper=10)
    if outlier_count:
        print(f"[✓] {outlier_count} adet bedrooms outlier'ı 10'a indirildi")

    print(f"[✓] Temizleme tamamlandı: {df.shape[1]} sütun kaldı")
    return df


# ─── 3. Feature Engineering ───────────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Alan bilgisi, bina yaşı ve konum özelliklerinden
    anlamlı bileşik değişkenler türetir.

    Korelasyon tabanlı luxury_score ağırlıkları (Pearson, ham veri):
      sqft_living=0.702, grade=0.667, view=0.397, waterfront=0.266
    """
    df = df.copy()

    # Bina yaşı — proje tanımında açıkça isteniyor
    df["house_age"] = df["sale_year"] - df["yr_built"]

    # Renovasyon özellikleri
    df["renovated"]         = (df["yr_renovated"] > 0).astype(int)
    df["years_since_reno"]  = np.where(
        df["yr_renovated"] > 0,
        df["sale_year"] - df["yr_renovated"],
        0
    )
    df = df.drop(columns=["yr_built", "yr_renovated"])

    # Alan oranları
    df["sqft_per_bedroom"]   = df["sqft_living"] / (df["bedrooms"].clip(lower=1))
    df["sqft_living_ratio"]  = df["sqft_living"] / (df["sqft_living15"].clip(lower=1))
    df["basement_ratio"]     = df["sqft_basement"] / (df["sqft_living"].clip(lower=1))

    # Korelasyon ağırlıklı lüks skoru
    df["luxury_score"] = (
        df["grade"]      * 0.667 +
        df["view"]       * 0.397 +
        df["waterfront"] * 0.266
    )

    new_cols = ["house_age", "renovated", "years_since_reno",
                "sqft_per_bedroom", "sqft_living_ratio",
                "basement_ratio", "luxury_score"]
    print(f"[✓] Feature engineering: {new_cols}")
    return df


# ─── 4. Log Transform (Hedef Değişken) ────────────────────────────────────────
def log_transform_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fiyat dağılımı sağa çarpık (skew=4.02). Log transform uygular.
    log1p kullanılır: log(1 + price) → negatif değerlere karşı güvenli.

    Kaynak: Malpezzi (2003) — hedonic pricing modellerinde standart.
    """
    df = df.copy()
    df[TARGET_COL] = np.log1p(df[TARGET_COL])
    print(f"[✓] Log transform uygulandı: fiyat aralığı "
          f"{df[TARGET_COL].min():.2f} — {df[TARGET_COL].max():.2f}")
    return df


# ─── 5. Özellik / Hedef Ayrımı ────────────────────────────────────────────────
def split_features_target(df: pd.DataFrame):
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL].values.astype(np.float32)
    print(f"[✓] Özellik sayısı: {X.shape[1]}  |  Hedef: log({TARGET_COL})")
    return X, y


# ─── 6. Train / Val / Test Bölme ──────────────────────────────────────────────
def split_data(X, y, random_state: int = 42):
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=random_state
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=random_state
    )
    print(f"[✓] Train: {len(X_train):,}  |  Val: {len(X_val):,}  |  Test: {len(X_test):,}")
    return X_train, X_val, X_test, y_train, y_val, y_test


# ─── 7. Ölçeklendirme ─────────────────────────────────────────────────────────
def scale_features(X_train, X_val, X_test):
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s   = scaler.transform(X_val)
    X_test_s  = scaler.transform(X_test)
    joblib.dump(scaler, SCALER_PATH)
    print(f"[✓] X Scaler kaydedildi: {SCALER_PATH}")
    return X_train_s, X_val_s, X_test_s, scaler


def scale_target(y_train, y_val, y_test):
    """
    Log transform sonrası StandardScaler — model optimize edilmiş
    uzayda çalışır, gradyanlar dengeli kalır.
    """
    y_scaler = StandardScaler()
    y_train_s = y_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_val_s   = y_scaler.transform(y_val.reshape(-1, 1)).flatten()
    y_test_s  = y_scaler.transform(y_test.reshape(-1, 1)).flatten()
    joblib.dump(y_scaler, Y_SCALER_PATH)
    print(f"[✓] y Scaler kaydedildi: {Y_SCALER_PATH}")
    return y_train_s, y_val_s, y_test_s, y_scaler


# ─── 8. Kaydet ────────────────────────────────────────────────────────────────
def save_splits(X_train, X_val, X_test, y_train, y_val, y_test, feature_names):
    def to_df(X, y):
        d = pd.DataFrame(X, columns=feature_names)
        d[TARGET_COL] = y
        return d

    to_df(X_train, y_train).to_csv(os.path.join(PROCESSED_DIR, "train.csv"), index=False)
    to_df(X_val,   y_val  ).to_csv(os.path.join(PROCESSED_DIR, "val.csv"),   index=False)
    to_df(X_test,  y_test ).to_csv(os.path.join(PROCESSED_DIR, "test.csv"),  index=False)
    print(f"[✓] İşlenmiş dosyalar kaydedildi: {PROCESSED_DIR}/")


# ─── Ana Pipeline ──────────────────────────────────────────────────────────────
def run_preprocessing():
    print("\n========== VERİ ÖN İŞLEME ==========\n")

    df = load_data()
    df = clean_data(df)
    df = engineer_features(df)
    df = log_transform_target(df)

    X, y = split_features_target(df)
    feature_names = list(X.columns)

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    X_train_s, X_val_s, X_test_s, scaler   = scale_features(X_train, X_val, X_test)
    y_train_s, y_val_s, y_test_s, y_scaler = scale_target(y_train, y_val, y_test)

    save_splits(X_train_s, X_val_s, X_test_s, y_train_s, y_val_s, y_test_s, feature_names)

    print("\n========== ÖZET ==========")
    print(f"  Özellikler ({len(feature_names)}): {feature_names}")
    print(f"  Train: {X_train_s.shape}  |  Val: {X_val_s.shape}  |  Test: {X_test_s.shape}")
    raw_prices = np.expm1(y_train)
    print(f"  Ham fiyat aralığı: ${raw_prices.min():,.0f} — ${raw_prices.max():,.0f}")
    print("==========================\n")

    return feature_names, scaler, y_scaler


if __name__ == "__main__":
    run_preprocessing()
