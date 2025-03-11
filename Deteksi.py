from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load semua model dari file pickle
with open("all_models.pkl", "rb") as f:
    all_models = pickle.load(f)

# Fungsi untuk melakukan prediksi dengan model ensemble
def stacked_prediction(input_data, model_name):
    try:
        models = all_models[model_name]
        
        # Pastikan input data dalam bentuk numpy array 2D
        input_array = np.array(input_data).reshape(1, -1)

        # Prediksi probabilitas dengan model dasar
        xgb_proba = models["xgb_model"].predict_proba(input_array)
        rf_proba = models["rf_model"].predict_proba(input_array)

        # Gabungkan hasil probabilitas sebagai input ke meta model
        stacked_features = np.hstack((xgb_proba, rf_proba))

        # Pastikan jumlah fitur sesuai dengan meta model
        expected_features = models["meta_model"].n_features_in_
        if stacked_features.shape[1] != expected_features:
            return {"error": f"Feature mismatch for {model_name}: expected {expected_features}, got {stacked_features.shape[1]}"}

        # Prediksi final dengan meta model
        final_prediction = models["meta_model"].predict(stacked_features)[0]

        # Decode label jika ada label encoder
        if models.get("label_encoder"):
            final_prediction = models["label_encoder"].inverse_transform([final_prediction])[0]

        # Pastikan hasil dikonversi ke tipe dasar Python sebelum dikembalikan
        return int(final_prediction) if isinstance(final_prediction, np.integer) else final_prediction

    except Exception as e:
        return {"error": str(e)}

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        results = {}

        if "diabetes" in data:
            diabetes_input = [
                data["diabetes"]["Pregnancies"],
                data["diabetes"]["BS"],
                data["diabetes"]["BloodPressure"],
                data["diabetes"]["SkinThickness"],
                data["diabetes"]["BMI"],
                data["diabetes"]["Age"]
            ]
            diabetes_result = stacked_prediction(diabetes_input, "diabetes")
            if isinstance(diabetes_result, dict) and "error" in diabetes_result:
                return jsonify(diabetes_result), 400
            results["diabetes_prediction"] = diabetes_result

        if "hypertension" in data:
            hypertension_input = [
                data["hypertension"]["sex"],
                data["hypertension"]["Age"],
                data["hypertension"]["currentSmoker"],
                data["hypertension"]["cigsPerDay"],
                data["hypertension"]["BPMeds"],
                data["hypertension"]["diabetes"],
                data["hypertension"]["SystolicBP"],
                data["hypertension"]["DiastolicBP"],
                data["hypertension"]["BMI"],
                data["hypertension"]["Heartrate"],
                data["hypertension"]["BS"]
            ]
            hypertension_result = stacked_prediction(hypertension_input, "hypertension")
            if isinstance(hypertension_result, dict) and "error" in hypertension_result:
                return jsonify(hypertension_result), 400
            results["hypertension_prediction"] = hypertension_result

        if "maternal_health" in data:
            maternal_input = [
                data["maternal_health"]["Age"],
                data["maternal_health"]["SystolicBP"],
                data["maternal_health"]["DiastolicBP"],
                data["maternal_health"]["BS"],
                data["maternal_health"]["BodyTemp"],
                data["maternal_health"]["HeartRate"]
            ]
            maternal_result = stacked_prediction(maternal_input, "maternal_health")
            if isinstance(maternal_result, dict) and "error" in maternal_result:
                return jsonify(maternal_result), 400
            results["maternal_health_prediction"] = maternal_result

        return jsonify(results)

    except KeyError as e:
        return jsonify({"error": f"Missing key: {str(e)}"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
