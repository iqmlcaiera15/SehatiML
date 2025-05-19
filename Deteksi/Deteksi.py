from flask import Flask, request, jsonify
import pickle
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Load semua model dari file pickle
try:
    with open("all_models.pkl", "rb") as f:
        all_models = pickle.load(f)
    logger.info("Models loaded successfully")
except Exception as e:
    logger.error(f"Error loading models: {str(e)}")
    all_models = {}

# Fungsi untuk melakukan prediksi dengan model ensemble
def stacked_prediction(input_data, model_name):
    try:
        if model_name not in all_models:
            logger.error(f"Model '{model_name}' not found in loaded models")
            return {"error": f"Model '{model_name}' not found"}
            
        models = all_models[model_name]
        logger.debug(f"Predicting with {model_name} model")
        
        # Pastikan input data dalam bentuk numpy array 2D
        input_array = np.array(input_data).reshape(1, -1)
        logger.debug(f"Input shape: {input_array.shape}, values: {input_array}")

        # Prediksi probabilitas dengan model dasar
        xgb_proba = models["xgb_model"].predict_proba(input_array)
        rf_proba = models["rf_model"].predict_proba(input_array)
        logger.debug(f"XGB probabilities shape: {xgb_proba.shape}")
        logger.debug(f"RF probabilities shape: {rf_proba.shape}")

        # Gabungkan hasil probabilitas sebagai input ke meta model
        stacked_features = np.hstack((xgb_proba, rf_proba))
        logger.debug(f"Stacked features shape: {stacked_features.shape}")

        # Pastikan jumlah fitur sesuai dengan meta model
        expected_features = models["meta_model"].n_features_in_
        if stacked_features.shape[1] != expected_features:
            logger.error(f"Feature mismatch for {model_name}: expected {expected_features}, got {stacked_features.shape[1]}")
            return {"error": f"Feature mismatch for {model_name}: expected {expected_features}, got {stacked_features.shape[1]}"}

        # Prediksi final dengan meta model
        final_prediction = models["meta_model"].predict(stacked_features)[0]
        logger.debug(f"Raw prediction: {final_prediction}")

        # Decode label jika ada label encoder
        if models.get("label_encoder"):
            final_prediction = models["label_encoder"].inverse_transform([final_prediction])[0]
            logger.debug(f"Decoded prediction: {final_prediction}")

        # Pastikan hasil dikonversi ke tipe dasar Python sebelum dikembalikan
        result = int(final_prediction) if isinstance(final_prediction, np.integer) else final_prediction
        logger.info(f"Final {model_name} prediction: {result}")
        return result

    except Exception as e:
        logger.error(f"Error in prediction for {model_name}: {str(e)}")
        return {"error": str(e)}

@app.route("/predictdeteksi", methods=["POST"])
def predict():
    try:
        data = request.json
        logger.info(f"Received request data: {data}")
        
        if not data:
            logger.warning("No JSON data received")
            return jsonify({"error": "No JSON data provided"}), 400
            
        results = {}

        if "diabetes" in data:
            logger.info("Processing diabetes prediction")
            try:
                diabetes_input = [
                    data["diabetes"]["Pregnancies"],
                    data["diabetes"]["BS"],
                    data["diabetes"]["BloodPressure"],
                    data["diabetes"]["SkinThickness"],
                    data["diabetes"]["BMI"],
                    data["diabetes"]["Age"]
                ]
                logger.debug(f"Diabetes input: {diabetes_input}")
                diabetes_result = stacked_prediction(diabetes_input, "diabetes")
                if isinstance(diabetes_result, dict) and "error" in diabetes_result:
                    logger.error(f"Diabetes prediction error: {diabetes_result['error']}")
                    results["diabetes_error"] = diabetes_result["error"]
                else:
                    results["diabetes_prediction"] = diabetes_result
            except KeyError as e:
                error_msg = f"Missing key in diabetes data: {str(e)}"
                logger.error(error_msg)
                results["diabetes_error"] = error_msg

        if "hypertension" in data:
            logger.info("Processing hypertension prediction")
            try:
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
                logger.debug(f"Hypertension input: {hypertension_input}")
                hypertension_result = stacked_prediction(hypertension_input, "hypertension")
                if isinstance(hypertension_result, dict) and "error" in hypertension_result:
                    logger.error(f"Hypertension prediction error: {hypertension_result['error']}")
                    results["hypertension_error"] = hypertension_result["error"]
                else:
                    results["hypertension_prediction"] = hypertension_result
            except KeyError as e:
                error_msg = f"Missing key in hypertension data: {str(e)}"
                logger.error(error_msg)
                results["hypertension_error"] = error_msg

        if "maternal_health" in data:
            logger.info("Processing maternal health prediction")
            try:
                maternal_input = [
                    data["maternal_health"]["Age"],
                    data["maternal_health"]["SystolicBP"],
                    data["maternal_health"]["DiastolicBP"],
                    data["maternal_health"]["BS"],
                    data["maternal_health"]["BodyTemp"],
                    data["maternal_health"]["HeartRate"]
                ]
                logger.debug(f"Maternal health input: {maternal_input}")
                maternal_result = stacked_prediction(maternal_input, "maternal_health")
                if isinstance(maternal_result, dict) and "error" in maternal_result:
                    logger.error(f"Maternal health prediction error: {maternal_result['error']}")
                    results["maternal_health_error"] = maternal_result["error"]
                else:
                    results["maternal_health_prediction"] = maternal_result
            except KeyError as e:
                error_msg = f"Missing key in maternal health data: {str(e)}"
                logger.error(error_msg)
                results["maternal_health_error"] = error_msg

        if not results:
            logger.warning("No predictions were made")
            return jsonify({"warning": "No valid prediction requests found in the payload"}), 200

        logger.info(f"Returning results: {results}")
        return jsonify(results)

    except Exception as e:
        error_msg = str(e)
        logger.error(f"Global error: {error_msg}")
        return jsonify({"error": error_msg}), 500

if __name__ == "__main__":
    logger.info("Starting Flask application")
    app.run(debug=True)