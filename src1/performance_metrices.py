import pandas as pd
import matplotlib.pyplot as plt
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# Create outputs folder
os.makedirs("outputs", exist_ok=True)

try:
    # Load processed data
    X_train = pd.read_csv("data/preprocessed/X_train.csv")
    X_test = pd.read_csv("data/preprocessed/X_test.csv")
    y_train = pd.read_csv("data/preprocessed/y_train.csv").values.ravel()
    y_test = pd.read_csv("data/preprocessed/y_test.csv").values.ravel()

    print("[INFO] Processed data loaded successfully")

except Exception as e:
    print(f"[ERROR] Data loading failed: {e}")
    exit()

# Train model
try:
    model = RandomForestClassifier(n_estimators=100, random_state=42)

    model.fit(X_train, y_train)

    print("[INFO] Model training completed")

except Exception as e:
    print(f"[ERROR] Model training failed: {e}")
    exit()

# Predictions
try:
    y_pred = model.predict(X_test)

except Exception as e:
    print(f"[ERROR] Prediction failed: {e}")
    exit()

# Calculate metrics
try:
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("\n===== PERFORMANCE METRICS =====")

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

except Exception as e:
    print(f"[ERROR] Metrics calculation failed: {e}")
    exit()

# Create graph
try:
    metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
    scores = [accuracy, precision, recall, f1]

    plt.figure(figsize=(8,5))

    bars = plt.bar(metrics, scores)

    plt.ylim(0, 1)

    plt.xlabel("Performance Metrics")
    plt.ylabel("Score")

    plt.title("Performance Metrics Interpretation")

    # Add values on bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2,
            yval + 0.02,
            f"{yval:.2f}",
            ha='center'
        )

    # Save graph
    plt.savefig("outputs/performance_metrics.png")

    print("[INFO] Performance metrics graph saved successfully")

    plt.show()

except Exception as e:
    print(f"[ERROR] Graph generation failed: {e}")