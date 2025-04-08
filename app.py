from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import os

app = Flask(__name__)
CORS(app)

# Load model and vectorizer from local model folder
model = joblib.load("model/model.pkl")
vectorizer = joblib.load("model/vectorizer.pkl")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    symptoms = data.get("symptoms", "")

    if not symptoms:
        return jsonify({"error": "No symptoms provided"}), 400

    symptoms_vec = vectorizer.transform([symptoms])
    prediction = model.predict(symptoms_vec)[0]

    return jsonify({"predicted_disease": prediction})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
