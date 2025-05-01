from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load model SVM yang sudah disimpan
model_path = r"svm_model_depression.pkl"  # Gunakan 'r' di depan string untuk menghindari error path 
try:
    with open(model_path, "rb") as file:
        model = pickle.load(file)
except Exception as e:
    print(f"Error loading model: {e}")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data or "features" not in data:
            return jsonify({"error": "Invalid request format. 'features' field is required"}), 400

        features = data["features"]

        # Pastikan data adalah list angka
        if not isinstance(features, list) or not all(isinstance(x, (int, float)) for x in features):
            return jsonify({"error": "Features must be a list of numbers"}), 400

        data_array = np.array([features]).reshape(1, -1)  # Ubah ke array numpy

        prediction = model.predict(data_array)  # Prediksi dengan model SVM
        return jsonify({"prediction": int(prediction[0])})
    
    except Exception as e:
        print(f"Prediction Error: {e}")
        return jsonify({"error": str(e)}), 500  # Kembalikan error dengan kode 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
