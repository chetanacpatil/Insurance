from flask import Flask, request, jsonify
import numpy as np
import joblib

# Load the trained model
model = joblib.load("insurancenw.pkl")

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return "Health Insurance Cost Prediction API"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from request
        data = request.get_json()

        # Extract features
        age = data['age']
        AnyTransplant = data['AnyTransplant']
        AnyChronicDiseases = data['AnyChronicDiseases']
        Historyofcancerinfamily = data['Historyofcancerinfamily']
        Numberofmajorsurgeries = data['Numberofmajorsurgeries']
        Height = data['Height']
        Weight = data['Weight']

        # Calculate BMI
        bmi = Weight / ((Height / 100) ** 2)

        # Prepare feature array
        features = np.array([[age, AnyTransplant, AnyChronicDiseases, Historyofcancerinfamily, Numberofmajorsurgeries, bmi]])

        # Make prediction
        prediction = model.predict(features)[0]

        # Return response
        return jsonify({'estimated_premium': float(prediction)})

    except Exception as e:
        return jsonify({'error': str(e)})

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)