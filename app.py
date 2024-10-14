from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd

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
    prediction = svm_model.predict(feature_values_scaled)
    decoded_prediction = label_encoder.inverse_transform(prediction)

    prediction_result = {
        'age': age,
        'heart_rate': heart_rate,
        'spo2': spo2,
        'temperature': temperature,
        'accelerometer': accelerometer,
        'prediction': decoded_prediction[0]
    }
    
    file_path = 'child_health_predict.csv'
    try:
        df = pd.read_csv(file_path)
        new_df = pd.DataFrame([prediction_result])
        df = pd.concat([df, new_df], ignore_index=True)
    except FileNotFoundError:
        df = pd.DataFrame([prediction_result])
    df.to_csv(file_path, index=False)
    
    return jsonify({'trạng thái dự đoán': decoded_prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)