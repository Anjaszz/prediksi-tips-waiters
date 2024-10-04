from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
from flask_cors import CORS

app = Flask(__name__)

# Konfigurasi CORS untuk mengizinkan akses dari port 5500
CORS(app, resources={r"/*": {"origins": "*"}})

# Load model dan scaler
model = load_model('model_tip_predictor.h5')
scaler = StandardScaler()

# Tambahkan handler untuk preflight request (OPTIONS)
@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    if request.method == 'OPTIONS':
        # Handling preflight request untuk CORS
        response = jsonify({'status': 'ok'})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Methods', 'POST, OPTIONS')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        return response

    elif request.method == 'POST':
        # Ambil data dari permintaan
        data = request.json
        total_bill = float(data['total_bill'])
        gender = int(data['gender'])
        smoker = int(data['smoker'])
        day = int(data['day'])
        time = int(data['time'])
        people = int(data['people'])

        # Preprocess input
        input_data = np.array([[total_bill, gender, smoker, day, time, people]])
        scaled_data = scaler.transform(input_data)  # Normalisasi input

        # Prediksi
        prediction = model.predict(scaled_data)
        return jsonify({'predicted_tip': prediction[0][0]})

if __name__ == '__main__':
    app.run(debug=True)
