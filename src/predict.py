import joblib
import numpy as np
import os
import pandas as pd

EXPECTED_FEATURES=30


try:
    if not os.path.exists("models/model.pkl"):
        raise FileNotFoundError("Model File not found")

    if not os.path.exists("models/scaler.pkl"):
        raise FileNotFoundError("Scaler File not found")
    
    model=joblib.load("models/model.pkl")
    scaler=joblib.load("models/scaler.pkl")
    
    print("[INFO] Models and Scaler loaded successfully")

except Exception as e:
    print(f"[ERROR] Initialization Failed :{e}")
    exit()

def validate_input(data):
    if data is None:
        return "Input Data is None"
    
    if not isinstance(data,(list,tuple)):
        return "Input must be list or tuple"
    
    if len (data)!=EXPECTED_FEATURES:
        return f"Expected {EXPECTED_FEATURES} features, got {len(data)}"
    for i , val in enumerate(data):
        if not isinstance(val,(int,float)):
            return f"Invalid value at index {i}: must be numeric"
    return None
def predict_transaction(data):
    try:
        #  Input validation
        validation_error = validate_input(data)
        if validation_error:
            return {
                "status": "error",
                "message": validation_error
            }

        # Convert input
        # Load feature names
        feature_names = joblib.load("models/feature_names.pkl")
        # Create DataFrame with correct column names
        features = pd.DataFrame([data], columns=feature_names)
        
       

        #  Scaling
        try:
            features_scaled = scaler.transform(features)
            features_scaled = pd.DataFrame(features_scaled, columns=feature_names)
        except Exception as e:
            return {
                "status": "error",
                "message": f"Scaling failed: {e}"
            }

        #  Prediction
        try:
            prediction = model.predict(features_scaled)[0]
            probability = model.predict_proba(features_scaled)[0][1]
        except Exception as e:
            return {
                "status": "error",
                "message": f"Model prediction failed: {e}"
            }

        #  Risk scoring
        if probability > 0.8:
            risk_level = "HIGH"
        elif probability > 0.5:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        return {
            "status": "success",
            "fraud": bool(prediction),
            "risk_score": float(probability),
            "risk_level": risk_level
        }

    except Exception as e:
        return {
            "status": "error",
            "message": f"Unexpected error: {e}"
        }

if __name__=="__main__":
    print("\n--- Positive Case ---")
    print(predict_transaction([0] * 30))

    print("\n--- Negative Case: Wrong length ---")
    print(predict_transaction([0] * 10))

    print("\n--- Negative Case: Non-numeric ---")
    print(predict_transaction([0] * 29 + ["abc"]))

    print("\n--- Negative Case: None input ---")
    print(predict_transaction(None))
