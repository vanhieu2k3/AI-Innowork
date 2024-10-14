from flask import Flask, request, jsonify
import joblib
import numpy as np

svm_model = joblib.load('svm_model.pkl')
scaler = joblib.load('scaler.pkl') 
label_encoder = joblib.load('label_encoder.pkl')

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    age = data['age']
    heart_rate = data['heart_rate']
    spo2 = data['spo2']
    temperature = data['temperature']
    accelerometer = data['accelerometer']
    
    feature_values = np.array([[age, heart_rate, spo2, temperature, accelerometer]])
    feature_values_scaled = scaler.transform(feature_values)
    predictions = svm_model.predict(feature_values_scaled)
    decoded_predictions = label_encoder.inverse_transform(predictions)
    
    return jsonify({'trạng thái dự đoán': decoded_predictions[0]})

if __name__ == '__main__':
    app.run(debug=True)