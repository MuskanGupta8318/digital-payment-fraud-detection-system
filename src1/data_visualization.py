import pandas as pd
import matplotlib.pyplot as plt
import os

# Create outputs folder
os.makedirs("outputs", exist_ok=True)

# Load dataset
df = pd.read_csv("data/creditcard.csv")

# Count fraud and normal transactions
counts = df["Class"].value_counts()

# Plot graph
plt.figure(figsize=(6,5))
counts.plot(kind='bar')

plt.title("Fraud vs Non-Fraud Transactions")
plt.xlabel("Class")
plt.ylabel("Count")

plt.xticks([0,1], ["Normal", "Fraud"])

# Save graph
plt.savefig("outputs/fraud_distribution.png")

print("[INFO] Graph saved successfully")

plt.show()