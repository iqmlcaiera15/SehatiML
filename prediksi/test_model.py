import pandas as pd

# Path ke file debug_input_flask.csv
csv_path = 'debug_input_flask.csv'  # Pastikan path ini benar

# Baca file CSV
df = pd.read_csv(csv_path)

# Tampilkan semua kolom
print("\n🔎 Daftar Kolom:")
print(list(df.columns))

# Tampilkan isi 1 baris input
print("\n🔎 Data Baris Pertama:")
print(df.iloc[0])

# Cari fitur One-Hot aktif (yang nilainya 1)
one_hot_aktif = df.iloc[0][df.iloc[0] == 1]

print("\n🔎 Fitur One-Hot yang Aktif (nilai=1):")
print(one_hot_aktif)

# Cek fitur numerik penting
print("\n🔎 Nilai Fitur Numerik:")
for feature in ["Usia Ibu", "Tekanan Darah", "Riwayat Persalinan", "Posisi Janin"]:
    print(f"{feature}: {df.iloc[0][feature]}")
