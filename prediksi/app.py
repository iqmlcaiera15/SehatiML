from flask import Flask, request, jsonify
from collections import OrderedDict
import pickle
import pandas as pd
import logging
import os
import json
import numpy as np
from lime.lime_tabular import LimeTabularExplainer

# Optional: Load .env
try:
    from dotenv import load_dotenv
    load_dotenv()
except:
    pass

app = Flask(__name__)

# 🔹 Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# 🔹 Load model dan buat ulang explainer saat startup
try:
    with open("random_forest_model.pkl", "rb") as model_file:
        model = pickle.load(model_file)

    with open("feature_columns.pkl", "rb") as feature_file:
        feature_columns = pickle.load(feature_file)

    # --- PERUBAHAN UTAMA: Buat ulang explainer, jangan load dari file ---
    # 1. Load data training (X_train) yang digunakan untuk membuat explainer
    with open("X_train.pkl", "rb") as f:
        X_train = pickle.load(f)

    # 2. Buat objek LimeTabularExplainer secara dinamis
    explainer = LimeTabularExplainer(
        training_data=X_train.values,
        feature_names=feature_columns,
        class_names=['Normal', 'Caesar'], # Sesuaikan dengan kelas model Anda
        mode='classification'
    )
    # -----------------------------------------------------------------

    logging.info("Model, fitur, dan explainer berhasil dimuat/dibuat.")
except Exception as e:
    logging.error(f"Gagal load model/explainer: {e}")
    raise e

# 🔹 Mapping input (semua string → integer)
mapping_tekanan_darah = {"normal": 0, "rendah": 1, "tinggi": 2}
mapping_riwayat_persalinan = {"tidak ada": 0, "normal": 1, "caesar": 2}
mapping_posisi_janin = {"normal": 0, "lintang": 1, "sungsang": 2}
mapping_hasil_prediksi = {0: "Normal", 1: "Caesar"}

def transform_input(data):
    try:
        encoded_input = {col: 0 for col in feature_columns}
        
        rki = str(data.get('riwayat_kesehatan_ibu', 'normal')).strip().title()
        if rki.lower() == "normal":
            rki = "Tidak Ada"
        kkj = str(data.get('kondisi_kesehatan_janin', 'normal')).strip().title()

        riwayat_col = f"Riwayat Kesehatan Ibu_{rki}"
        kondisi_col = f"Kondisi Kesehatan Janin_{kkj}"
        if riwayat_col in encoded_input:
            encoded_input[riwayat_col] = 1
        if kondisi_col in encoded_input:
            encoded_input[kondisi_col] = 1

        tekanan_darah = mapping_tekanan_darah.get(str(data["tekanan_darah"]).strip().lower())
        riwayat_persalinan = mapping_riwayat_persalinan.get(str(data["riwayat_persalinan"]).strip().lower())
        posisi_janin = mapping_posisi_janin.get(str(data["posisi_janin"]).strip().lower())

        if tekanan_darah is None or riwayat_persalinan is None or posisi_janin is None:
            raise ValueError("Input tidak valid (tekanan_darah, riwayat_persalinan, posisi_janin)")

        encoded_input.update({
            "Usia Ibu": int(data["usia_ibu"]),
            "Tekanan Darah": tekanan_darah,
            "Riwayat Persalinan": riwayat_persalinan,
            "Posisi Janin": posisi_janin
        })

        return pd.DataFrame([encoded_input]).reindex(columns=feature_columns, fill_value=0), None
    except Exception as e:
        logging.error(f"Gagal transformasi input: {e}")
        return None, str(e)

def interpret_main_cause(model, input_df, explainer, num_features=6):
    try:
        instance = input_df.values[0]
        explanation = explainer.explain_instance(instance, model.predict_proba, num_features=num_features)
        
        positive_features = [(feat, weight) for feat, weight in explanation.as_list() if weight > 0]

        if positive_features:
            top_feature = sorted(positive_features, key=lambda x: x[1], reverse=True)[0][0]
            nama_mentah = top_feature.split()[0].replace("_", " ")

            nama_fitur_mapping = {
                "Usia": "Usia Ibu", "Tekanan": "Tekanan Darah", "Riwayat": "Riwayat Persalinan",
                "Posisi": "Posisi Janin", "Kondisi": "Kondisi Kesehatan Janin",
                "Riwayat Kesehatan": "Riwayat Kesehatan Ibu"
            }
            for kunci, nama_bersih in nama_fitur_mapping.items():
                if nama_mentah.startswith(kunci):
                    return nama_bersih
            return nama_mentah.capitalize()
        else:
            return "Tidak diketahui (tidak ada fitur positif)"
            
    except Exception as e:
        # Kode dikembalikan ke versi produksi
        logging.error(f"Gagal interpretasi LIME: {e}")
        return "Tidak diketahui"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        logging.info(f"Data masuk:\n{json.dumps(data, indent=2, ensure_ascii=False)}")

        required_fields = [
            "usia_ibu", "tekanan_darah", "riwayat_persalinan",
            "posisi_janin", "riwayat_kesehatan_ibu", "kondisi_kesehatan_janin"
        ]
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Field {field} tidak boleh kosong"}), 400

        transformed_input, error = transform_input(data)
        if error:
            return jsonify({"error": error}), 400

        prediksi = model.predict(transformed_input)[0]
        hasil_prediksi = mapping_hasil_prediksi.get(prediksi, "unknown")

        probas = model.predict_proba(transformed_input)[0]
        pred_idx = list(mapping_hasil_prediksi.keys())[list(mapping_hasil_prediksi.values()).index(hasil_prediksi)]
        confidence = float(probas[pred_idx])
        confidence_percent = round(confidence * 100)

        response = OrderedDict([
            ("status", "success"),
            ("message", "Prediksi metode persalinan berhasil"),
            ("hasil_prediksi", hasil_prediksi),
            ("confidence", confidence_percent)
        ])

        if hasil_prediksi.lower() == "caesar":
            faktor = interpret_main_cause(model, transformed_input, explainer, num_features=6)
            response["faktor"] = faktor

        return jsonify(response), 200

    except Exception as e:
        logging.error(f"Error saat prediksi: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug_mode = os.environ.get("FLASK_DEBUG", "true").lower() == "true"
    app.run(debug=debug_mode, host="0.0.0.0", port=port)
