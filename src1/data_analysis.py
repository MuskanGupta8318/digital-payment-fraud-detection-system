import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# =========================
# LOAD DATA
# =========================
df = pd.read_csv("data/creditcard.csv", engine="python")

sns.set_style("whitegrid")


# =========================
# 1. CLASS DISTRIBUTION
# =========================
plt.figure(figsize=(6,5))

sns.countplot(x='Class', data=df)

plt.title("Class Distribution (Fraud vs Legitimate)")
plt.xticks([0,1], ["Legitimate", "Fraud"])
plt.ylabel("Count")

plt.tight_layout()
plt.show()


# =========================
# 2. TRANSACTION AMOUNT DISTRIBUTION
# =========================
plt.figure(figsize=(8,5))

sns.histplot(df['Amount'], bins=50, kde=True)

plt.title("Transaction Amount Distribution")
plt.xlabel("Amount")
plt.ylabel("Frequency")

plt.tight_layout()
plt.show()


# =========================
# 3. TRANSACTION TIME DISTRIBUTION
# =========================
plt.figure(figsize=(8,5))

sns.histplot(df['Time'], bins=50, kde=True)

plt.title("Transaction Time Distribution")
plt.xlabel("Time")
plt.ylabel("Frequency")

plt.tight_layout()
plt.show()



plt.figure(figsize=(10,5))

sns.histplot(df['Time'], bins=100)

plt.title("Transaction Frequency Over Time")
plt.xlabel("Time (seconds)")
plt.ylabel("Number of Transactions")

plt.tight_layout()
plt.show()


# =========================
# 4. OUTLIER DETECTION (BOXPLOT - RAW)
# =========================
plt.figure(figsize=(8,5))

sns.boxplot(x=df['Amount'])

plt.title("Outlier Detection - Transaction Amount (Raw)")
plt.xlabel("Amount")

plt.tight_layout()
plt.show()


# =========================
# 5. OUTLIER DETECTION (BOXPLOT - LOG SCALE)
# =========================
plt.figure(figsize=(8,5))

sns.boxplot(x=np.log1p(df['Amount']))

plt.title("Outlier Detection - Transaction Amount (Log Scale)")
plt.xlabel("Log(Amount + 1)")

plt.tight_layout()
plt.show()


# =========================
# 6. OUTLIERS BY CLASS
# =========================
plt.figure(figsize=(8,5))

sns.boxplot(x='Class', y='Amount', data=df)

plt.title("Outliers by Class (Fraud vs Legitimate)")
plt.xticks([0,1], ["Legitimate", "Fraud"])
plt.ylabel("Amount")

plt.tight_layout()
plt.show()


# =========================
# 7. IQR OUTLIER DETECTION (STATISTICAL)
# =========================
Q1 = df['Amount'].quantile(0.25)
Q3 = df['Amount'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = df[(df['Amount'] < lower_bound) | (df['Amount'] > upper_bound)]

print("\n================ OUTLIER SUMMARY ================")
print("Total Records:", len(df))
print("Q1:", Q1)
print("Q3:", Q3)
print("IQR:", IQR)
print("Lower Bound:", lower_bound)
print("Upper Bound:", upper_bound)
print("Total Outliers:", len(outliers))
print("Outlier Percentage:", round(len(outliers)/len(df)*100, 2), "%")


# =========================
# 8. OUTLIER VISUALIZATION (SCATTER)
# =========================
plt.figure(figsize=(8,5))

plt.scatter(outliers.index, outliers['Amount'], color='red', s=10)

plt.title("Detected Outliers (IQR Method)")
plt.xlabel("Index")
plt.ylabel("Amount")

plt.tight_layout()
plt.show()