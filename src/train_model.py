import pandas as pd
import joblib 
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix

def log(msg):
    print(f"[INFO] {msg}")

def error(msg):
    print(f"[ERROR] {msg}")

try:
    X_train=pd.read_csv("data/preprocessed/X_train.csv")
    X_test=pd.read_csv("data/preprocessed/X_test.csv")
    y_train=pd.read_csv("data/preprocessed/y_train.csv")
    y_test=pd.read_csv("data/preprocessed/y_test.csv")
    log("Processed data loaded successfully")
except Exception as e:
    error(f"Processed data not loaded successfully")
    exit()

try:
    if X_train.empty or X_test.empty:
        raise ValueError("Training or Testing Data is empty")
    if y_train.empty or y_test.empty:
        raise ValueError("Target Data is empty")
except Exception as e:
    error(e)
    exit()

#Train Model

try:
    model=RandomForestClassifier(n_estimators=100,random_state=42)
    model.fit(X_train,y_train.values.ravel())
    log("Model Training Completed")

except Exception as e:
    error(e)
    exit()

#Prediction

try :
    y_pred=model.predict(X_test)

except Exception as e:
    error(f"Prediction not successful {e}")
    exit()

#Evaluation
try:
    accuracy=accuracy_score(y_test,y_pred)
    precision=precision_score(y_test,y_pred,zero_division=0)
    recall=recall_score(y_test,y_pred,zero_division=0)
    f1=f1_score(y_test,y_pred,zero_division=0)
    confusionmatrix=confusion_matrix(y_test,y_pred)
    print("\n Model Evaluation:")
    print("Accuracy :", accuracy)
    print("Precision:", precision)
    print("Recall   :", recall)
    print("F1 Score :", f1)

    print("\nConfusion Matrix:\n", confusionmatrix)

except Exception as e:
    error(f"Evaluation failed: {e}")
    exit()

 #Save Model
 
try:
    os.makedirs("models", exist_ok=True)

    joblib.dump(model, "models/model.pkl")

    log("Model saved successfully!")

except Exception as e:
    error(f"Model saving failed: {e}")
    exit()

log("Training  completed successfully!")