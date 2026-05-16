import pandas as pd
import matplotlib.pyplot as plt
import os

# Create outputs folder
os.makedirs("outputs", exist_ok=True)

try:
    # Load dataset
    df = pd.read_csv("data/creditcard.csv")

    print("[INFO] Dataset loaded successfully")

except Exception as e:
    print(f"[ERROR] Dataset loading failed: {e}")
    exit()

try:
    # Select important features only
    important_features = ["Time", "Amount"]

    # Add few V columns if available
    for col in ["V1", "V2", "V3", "V4"]:
        if col in df.columns:
            important_features.append(col)

    # Create plots
    plt.figure(figsize=(14,8))

    for i, feature in enumerate(important_features, 1):

        plt.subplot(2, 3, i)

        plt.hist(df[feature], bins=30)

        plt.title(feature)
        plt.xlabel("Value")
        plt.ylabel("Frequency")

    plt.tight_layout()

    # Save graph
    plt.savefig("outputs/feature_analysis.png")

    print("[INFO] Feature analysis graph saved successfully")

    plt.show()

except Exception as e:
    print(f"[ERROR] Feature analysis failed: {e}")