from flask import Flask,request,jsonify
import joblib
import pandas as pd
import os
app=Flask(__name__)
EXPECTED_FEATURES=30
try:
    model=joblib.load("models/model.pkl")
    scaler=joblib.load("models/scaler.pkl")
    features_name=joblib.load("models/feature_names.pkl")

except Exception as e:
    print(f"Error in Loading Model {e}")
    exit()

#Validation

def validate_input(data):
    if data is None:
        return "Input Data is None"
    
    if not isinstance(data,list):
        return "Input must be list"
    
    if len (data)!=EXPECTED_FEATURES:
        return f"Expected {EXPECTED_FEATURES} features,got{len(data)}"
    
    for i ,val in enumerate (data):
        if not isinstance (val,(int,float)):
            return f"Invalid value at index {i} must be numeric"
    
    return None

#Prediction API
@app.route("/predict",methods=["POST"])
def predict():
    try:
        data=request.json.get("data")
        #Validate
        error=validate_input(data)
        if error:
            return jsonify ({"status":"error","message":error})
        #Convert to Data Frame
        features=pd.DataFrame([data],columns=features_name)
        #Scale
        features_scaled=scaler.transform(features)
        features_scaled=pd.DataFrame(features_scaled,columns=features_name)
        
        #Prediction
        prediction=model.predict(features_scaled)[0]
        probability=model.predict_proba(features_scaled)[0][1]

        #Risk 
        if probability>0.8:
            risk= "HIGH"
        elif probability>0.5:
            risk="MEDIUM"
        else:
            risk="LOW"
        
        return jsonify({
            "status":"success",
            "fraud": bool(prediction),
            "risk_score": float(probability),
            "risk_level": risk
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        })


# ==============================
# Run Server
# ==============================
if __name__ == "__main__":
    app.run(debug=True)

