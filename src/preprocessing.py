import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# =========================
# CREATE FOLDERS
# =========================
os.makedirs("outputs", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)

# =========================
# LOAD DATASET
# =========================
file_path = "data/creditcard.csv"
df = pd.read_csv(file_path)

print("Dataset Loaded Successfully")
print("Dataset Shape:", df.shape)

# =========================
# MISSING VALUES
# =========================
missing = df.isnull().sum()

plt.figure(figsize=(10,5))
missing.plot(kind='bar')
plt.title("Missing Values Analysis")
plt.xlabel("Features")
plt.ylabel("Missing Values Count")
plt.tight_layout()
plt.savefig("outputs/missing_values.png")
plt.close()

# =========================
# FRAUD DISTRIBUTION
# =========================
plt.figure(figsize=(6,5))
df["Class"].value_counts().plot(kind='bar')
plt.title("Fraud vs Non-Fraud Transactions")
plt.xlabel("Class")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("outputs/fraud_vs_nonfraud.png")
plt.close()

# =========================
# CORRELATION HEATMAP
# =========================
plt.figure(figsize=(12,10))
sns.heatmap(df.corr(), cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.savefig("outputs/correlation_heatmap.png")
plt.close()

# =========================
# TIME-BASED TRANSACTIONS (NEW)
# =========================
df["Hour"] = (df["Time"] // 3600)

plt.figure(figsize=(10,5))
df.groupby("Hour")["Class"].count().plot()

plt.title("Transaction Frequency Over Time (Hours)")
plt.xlabel("Hour")
plt.ylabel("Number of Transactions")
plt.tight_layout()
plt.savefig("outputs/time_based_transactions.png")
plt.close()

# =========================
# OUTLIER DETECTION (IQR) (NEW)
# =========================
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
outlier_counts = {}

for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    outliers = df[(df[col] < lower) | (df[col] > upper)]
    outlier_counts[col] = len(outliers)

plt.figure(figsize=(12,6))
plt.bar(outlier_counts.keys(), outlier_counts.values())
plt.xticks(rotation=90)
plt.title("Outlier Detection using IQR")
plt.ylabel("Outlier Count")
plt.tight_layout()
plt.savefig("outputs/outlier_iqr.png")
plt.close()

# =========================
# FEATURE VISUALIZATION (NEW)
# =========================
df[["V1", "V2", "V3", "Amount"]].hist(figsize=(10,6), bins=30)
plt.suptitle("Feature Distribution Visualization")
plt.tight_layout()
plt.savefig("outputs/feature_visualization.png")
plt.close()

# =========================
# BALANCING DATASET
# =========================
fraud = df[df["Class"] == 1]
normal = df[df["Class"] == 0]

normal_sample = normal.sample(n=len(fraud), random_state=42)

df_balanced = pd.concat([fraud, normal_sample])

# =========================
# SPLIT FEATURES/TARGET
# =========================
X = df_balanced.drop("Class", axis=1)
y = df_balanced["Class"]

# =========================
# TRAIN TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =========================
# TRAIN-TEST PIE CHART
# =========================
plt.figure(figsize=(6,6))
plt.pie(
    [len(X_train), len(X_test)],
    labels=["Training Data", "Testing Data"],
    autopct='%1.1f%%'
)
plt.title("Train-Test Data Split")
plt.savefig("outputs/train_test_split.png")
plt.close()

# =========================
# FEATURE SCALING
# =========================
scaler = StandardScaler()

before_scaling = X_train.iloc[:,0].values[:100]

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns)

after_scaling = X_train_scaled.iloc[:,0].values[:100]

plt.figure(figsize=(10,5))
plt.plot(before_scaling, label="Before Scaling")
plt.plot(after_scaling, label="After Scaling")
plt.title("Feature Scaling and Normalization")
plt.xlabel("Samples")
plt.ylabel("Values")
plt.legend()
plt.tight_layout()
plt.savefig("outputs/feature_scaling.png")
plt.close()

# =========================
# SAVE PROCESSED DATA
# =========================
X_train_scaled.to_csv("data/processed/X_train.csv", index=False)
X_test_scaled.to_csv("data/processed/X_test.csv", index=False)
y_train.to_csv("data/processed/y_train.csv", index=False)
y_test.to_csv("data/processed/y_test.csv", index=False)

print("Preprocessing Completed Successfully")
# Save scaler
joblib.dump(scaler, "models/scaler.pkl")

# Save feature names
joblib.dump(list(X.columns), "models/feature_names.pkl")