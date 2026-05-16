import pandas as pd
import matplotlib.pyplot as plt
import os
import joblib

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc
)

# =========================
# LOG FUNCTIONS
# =========================

def log(message):
    print(f"[INFO] {message}")

def error(message):
    print(f"[ERROR] {message}")

# =========================
# CREATE FOLDERS
# =========================

os.makedirs("outputs", exist_ok=True)
os.makedirs("models", exist_ok=True)

# =========================
# LOAD PROCESSED DATA
# =========================

try:
    X_train = pd.read_csv("data/processed/X_train.csv")
    X_test = pd.read_csv("data/processed/X_test.csv")

    y_train = pd.read_csv(
        "data/processed/y_train.csv"
    ).values.ravel()

    y_test = pd.read_csv(
        "data/processed/y_test.csv"
    ).values.ravel()

    log("Processed data loaded successfully")

except Exception as e:
    error(f"Unable to load processed data: {e}")
    exit()

# =========================
# MODEL TRAINING
# =========================

try:
    model = RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",
        random_state=42
    )

    model.fit(X_train, y_train)

    log("Random Forest model training completed")

except Exception as e:
    error(f"Model training failed: {e}")
    exit()

# =========================
# SAVE MODEL
# =========================

try:
    joblib.dump(model, "models/model.pkl")

    log("Model saved successfully")

except Exception as e:
    error(f"Model saving failed: {e}")

# =========================
# PREDICTIONS
# =========================

try:
    y_pred = model.predict(X_test)

    y_prob = model.predict_proba(X_test)[:, 1]

    log("Prediction completed")

except Exception as e:
    error(f"Prediction failed: {e}")
    exit()

# =========================
# DEBUG OUTPUT
# =========================

print("\n===== ACTUAL VALUES =====")
print(pd.Series(y_test).value_counts())

print("\n===== PREDICTED VALUES =====")
print(pd.Series(y_pred).value_counts())

# =========================
# PERFORMANCE METRICS
# =========================

try:
    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    print("\n===== MODEL PERFORMANCE =====")

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

except Exception as e:
    error(f"Performance evaluation failed: {e}")

# =========================
# CONFUSION MATRIX
# =========================

try:
    cm = confusion_matrix(y_test, y_pred)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm
    )

    disp.plot()

    plt.title("Confusion Matrix")

    plt.savefig(
        "outputs/confusion_matrix.png"
    )

    plt.close()

    log("Confusion matrix saved")

except Exception as e:
    error(f"Confusion matrix generation failed: {e}")

# =========================
# ROC CURVE
# =========================

try:
    fpr, tpr, thresholds = roc_curve(
        y_test,
        y_prob
    )

    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6,5))

    plt.plot(
        fpr,
        tpr,
        label=f"AUC = {roc_auc:.2f}"
    )

    plt.plot([0,1], [0,1], 'r--')

    plt.xlabel("False Positive Rate")

    plt.ylabel("True Positive Rate")

    plt.title("ROC Curve")

    plt.legend()

    plt.savefig(
        "outputs/roc_curve.png"
    )

    plt.close()

    log("ROC curve saved")

except Exception as e:
    error(f"ROC curve generation failed: {e}")

# =========================
# PERFORMANCE METRICS GRAPH
# =========================

try:
    metrics = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1-Score"
    ]

    scores = [
        accuracy,
        precision,
        recall,
        f1
    ]

    plt.figure(figsize=(8,5))

    bars = plt.bar(metrics, scores)

    plt.ylim(0, 1)

    plt.xlabel("Metrics")

    plt.ylabel("Score")

    plt.title("Performance Metrics")

    for bar in bars:

        yval = bar.get_height()

        plt.text(
            bar.get_x() + bar.get_width()/2,
            yval + 0.02,
            f"{yval:.2f}",
            ha='center'
        )

    plt.savefig(
        "outputs/performance_metrics.png"
    )

    plt.close()

    log("Performance metrics graph saved")

except Exception as e:
    error(f"Performance metrics graph failed: {e}")

# =========================
# FEATURE IMPORTANCE GRAPH
# =========================

try:
    importance = model.feature_importances_

    feature_importance = pd.DataFrame({
        "Feature": X_train.columns,
        "Importance": importance
    })

    feature_importance = feature_importance.sort_values(
        by="Importance",
        ascending=False
    )

    plt.figure(figsize=(10,6))

    plt.barh(
        feature_importance["Feature"],
        feature_importance["Importance"]
    )

    plt.xlabel("Importance Score")

    plt.ylabel("Features")

    plt.title("Feature Importance Analysis")

    plt.gca().invert_yaxis()

    plt.tight_layout()

    plt.savefig(
        "outputs/feature_importance.png"
    )

    plt.close()

    log("Feature importance graph saved")

except Exception as e:
    error(f"Feature importance generation failed: {e}")

# =========================
# COMPLETED
# =========================

log("Training and evaluation completed successfully")