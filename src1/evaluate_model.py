from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import pandas as pd

def evaluate_model(model, X_test, y_test, X):

    # Prediction
    y_pred = model.predict(X_test)

    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)

    print("Accuracy:", accuracy)

    # =========================
    # FEATURE IMPORTANCE GRAPH
    # =========================

    importance = model.feature_importances_

    feature_importance = pd.DataFrame({
        "Feature": X.columns,
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

    plt.savefig("outputs/feature_importance.png")

    plt.show()

    print("Feature Importance Graph Generated")