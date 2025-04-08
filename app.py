from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib

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
    app.run(debug=True)
