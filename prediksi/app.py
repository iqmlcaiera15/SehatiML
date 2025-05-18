from flask import Flask, request, jsonify
from collections import OrderedDict
import pickle
import pandas as pd
import logging
import os
import json

app = Flask(__name__)

# 🔹 Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# 🔹 Load Model & Feature Columns
try:
    with open("random_forest_model.pkl", "rb") as model_file:
        model = pickle.load(model_file)

    with open("feature_columns.pkl", "rb") as feature_file:
        feature_columns = pickle.load(feature_file)

    logging.info("✅ Model dan fitur berhasil dimuat.")
except Exception as e:
    logging.error(f"❌ Gagal memuat model atau fitur: {e}")
    raise e

# 🔹 Mapping String ke Numerik
mapping_tekanan_darah = {"normal": 0, "rendah": 1, "tinggi": 2}
mapping_riwayat_persalinan = {"tidak ada": 0, "normal": 1, "caesar": 2}
mapping_posisi_janin = {"normal": 0, "lintang": 1, "sungsang": 2}

# 🔹 Mapping Numerik ke Output String
mapping_hasil_prediksi = {0: "normal", 1: "caesar"}

def transform_input(data):
    """
    Transformasi input JSON ke DataFrame yang sesuai dengan fitur model.
    """
    try:
        logging.info("📩 Transformasi input dimulai...")

        # 🔹 Inisialisasi semua fitur dengan nilai 0
        encoded_input = {col: 0 for col in feature_columns}

        # 🔹 One-Hot Encoding untuk kategori teks
        riwayat_col = f"Riwayat Kesehatan Ibu_{data.get('riwayat_kesehatan_ibu', 'normal')}"
        kondisi_col = f"Kondisi Kesehatan Janin_{data.get('kondisi_kesehatan_janin', 'normal')}"

        if riwayat_col in encoded_input:
            encoded_input[riwayat_col] = 1
        else:
            logging.warning(f"⚠️ Fitur {riwayat_col} tidak ditemukan di feature_columns.")

        if kondisi_col in encoded_input:
            encoded_input[kondisi_col] = 1
        else:
            logging.warning(f"⚠️ Fitur {kondisi_col} tidak ditemukan di feature_columns.")

        # 🔹 Konversi Kategorikal ke Numerik
        tekanan_darah = data["tekanan_darah"]
        riwayat_persalinan = data["riwayat_persalinan"]
        posisi_janin = data["posisi_janin"]

        if isinstance(tekanan_darah, str):
            tekanan_darah = mapping_tekanan_darah.get(tekanan_darah.lower())
        if isinstance(riwayat_persalinan, str):
            riwayat_persalinan = mapping_riwayat_persalinan.get(riwayat_persalinan.lower())
        if isinstance(posisi_janin, str):
            posisi_janin = mapping_posisi_janin.get(posisi_janin.lower())

        if tekanan_darah is None or riwayat_persalinan is None or posisi_janin is None:
            raise ValueError("Input tidak valid untuk tekanan_darah, riwayat_persalinan, atau posisi_janin")

        # 🔹 Masukkan Fitur Numerik
        encoded_input.update({
            "Usia Ibu": int(data["usia_ibu"]),
            "Tekanan Darah": tekanan_darah,
            "Riwayat Persalinan": riwayat_persalinan,
            "Posisi Janin": posisi_janin
        })

        # 🔹 Konversi ke DataFrame dengan urutan kolom yang benar
        transformed_df = pd.DataFrame([encoded_input]).reindex(columns=feature_columns, fill_value=0)

        logging.info(f"✅ Data setelah transformasi:\n{transformed_df}")
        return transformed_df, None
    except Exception as e:
        logging.error(f"❌ Gagal mengonversi input: {e}")
        return None, str(e)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        logging.info(f"📥 Data diterima dari client:\n{json.dumps(data, indent=2, ensure_ascii=False)}")

        # 🔹 Validasi Input
        required_fields = ["usia_ibu", "tekanan_darah", "riwayat_persalinan", "posisi_janin", "riwayat_kesehatan_ibu", "kondisi_kesehatan_janin"]
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Field {field} tidak boleh kosong"}), 400

        # 🔹 Transformasi Data
        transformed_input, error = transform_input(data)
        if error:
            return jsonify({"error": error}), 400

        # 🔹 Simpan data untuk debugging
        debug_path = os.path.join(os.getcwd(), "debug_input_flask.csv")
        transformed_input.to_csv(debug_path, index=False)
        logging.info(f"✅ Data input disimpan ke {debug_path}")

        # 🔹 Prediksi
        prediksi = model.predict(transformed_input)[0]
        hasil_prediksi = mapping_hasil_prediksi.get(prediksi, "unknown")
        logging.info(f"✅ Hasil prediksi model: {hasil_prediksi}")

        # 🔹 Respons JSON ke client
        response = OrderedDict([
            ("status", "success"),
            ("message", "Prediksi metode persalinan berhasil"),
            ("hasil_prediksi", hasil_prediksi)
        ])
        return app.response_class(
            response=json.dumps(response, ensure_ascii=False),
            status=200,
            mimetype="application/json"
        )

    except Exception as e:
        logging.error(f"❌ Terjadi exception dalam prediksi: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
