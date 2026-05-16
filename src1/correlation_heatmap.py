import seaborn as sns
import matplotlib.pyplot as plt
import os
import pandas as pd

os.makedirs("outputs", exist_ok=True)

try:
    df = pd.read_csv("data/creditcard.csv")

    print("[INFO] Dataset loaded successfully")
    print("[INFO] Dataset Shape:", df.shape)

    # Compute correlation matrix
    corr = df.corr()

    # Plot heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr, cmap="coolwarm", annot=False)

    plt.title("Correlation Heatmap of Features")

    plt.savefig("outputs/correlation_heatmap.png")
    print("[INFO] Correlation heatmap saved successfully")

    plt.show()

except Exception as e:
    print(f"[ERROR] Failed: {e}")