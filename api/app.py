from flask import Flask, request, jsonify
import pandas as pd
import joblib

model = joblib.load("models/model.pkl")
scaler = joblib.load("models/scaler.pkl")
feature_names = joblib.load("models/feature_names.pkl")

app = Flask(__name__)

# =========================
# SAFE FEATURE BUILDER (WORKAROUND)
# =========================
def align_features(input_df):
    df = input_df.copy()

    # 🔥 WORKAROUND 1: If Hour missing, create it from Time
    if "Hour" in feature_names and "Hour" not in df.columns:
        if "Time" in df.columns:
            df["Hour"] = df["Time"] // 3600
        else:
            df["Hour"] = 0  # fallback safe value

    # 🔥 WORKAROUND 2: Add missing columns with 0
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0

    # Arrange exact order
    df = df[feature_names]

    return df


# =========================
# HOME
# =========================
@app.route("/")
def home():
    return jsonify({"message": "Fraud Detection API Running"})


# =========================
# PREDICT
# =========================
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        input_df = pd.DataFrame([data])

        # 🔥 FIX APPLIED HERE
        input_df = align_features(input_df)

        scaled = scaler.transform(input_df)

        prediction = model.predict(scaled)[0]
        probability = model.predict_proba(scaled)[0][1]

        # Risk logic
        if probability > 0.75:
            risk = "High Risk"
        elif probability > 0.3:
            risk = "Medium Risk"
        else:
            risk = "Low Risk"

        return jsonify({
            "prediction": int(prediction),
            "fraud_probability": round(float(probability), 4),
            "risk_level": risk
        })

    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    app.run(debug=True)