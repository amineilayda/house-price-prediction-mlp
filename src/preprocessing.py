import pandas as pd

# CSV dosyasını oku
df = pd.read_csv("data/raw/Housing.csv")

# İlk 5 satır
print("İLK 5 SATIR")
print(df.head())

# Veri seti boyutu
print("\nVERİ SETİ BOYUTU")
print(df.shape)

# Sütun isimleri
print("\nSÜTUNLAR")
print(df.columns)

# Veri tipleri
print("\nVERİ TİPLERİ")
print(df.dtypes)

# Eksik veri kontrolü
print("\nEKSİK VERİ KONTROLÜ")
print(df.isnull().sum())

# Sayısal istatistikler
print("\nİSTATİSTİKLER")
print(df.describe())