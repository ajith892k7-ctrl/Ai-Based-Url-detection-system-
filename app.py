"""
app.py
------
Flask web app for the Phishing URL Detection System.
Run with: python3 app.py
Then open http://localhost:5000
"""

from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd

from feature_extraction import extract_features

app = Flask(__name__)

# Load trained model bundle
bundle = joblib.load("models/phishing_model.pkl")
model = bundle["model"]
model_type = bundle["type"]
columns = bundle["columns"]

scaler = None
if model_type == "lr":
    scaler = joblib.load("models/scaler.pkl")


def predict_url(url: str):
    features = extract_features(url)
    X = pd.DataFrame([features])[columns]  # keep column order consistent

    if scaler is not None:
        X = scaler.transform(X)

    pred = model.predict(X)[0]
    prob = model.predict_proba(X)[0][1]  # probability of being phishing

    return {
        "url": url,
        "prediction": "Phishing" if pred == 1 else "Legitimate",
        "confidence": round(float(prob if pred == 1 else 1 - prob) * 100, 2),
        "risk_score": round(float(prob) * 100, 2),
        "features": features,
    }


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    url = data.get("url", "").strip()
    if not url:
        return jsonify({"error": "No URL provided"}), 400
    result = predict_url(url)
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
