import pandas as pd
import os
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# =========================
# LOG FUNCTIONS
# =========================

def log(message):
    print(f"[INFO] {message}")

def error(message):
    print(f"[ERROR] {message}")

# =========================
# CREATE OUTPUT FOLDERS
# =========================

os.makedirs("outputs", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)

# =========================
# LOAD DATASET
# =========================

file_path = "data/creditcard.csv"

try:
    if not os.path.exists(file_path):
        raise FileNotFoundError("Dataset file not found")

    df = pd.read_csv(file_path)

    log("Dataset loaded successfully")

except Exception as e:
    error(f"Dataset loading failed: {e}")
    exit()

# =========================
# DATASET VALIDATION
# =========================

try:
    if df.empty:
        raise ValueError("Dataset is empty")

    log(f"Dataset shape: {df.shape}")

except Exception as e:
    error(e)
    exit()

# =========================
# TARGET COLUMN CHECK
# =========================

TARGET = "Class"

try:
    if TARGET not in df.columns:
        raise ValueError(f"{TARGET} column not found")

    log("Target column found")

except Exception as e:
    error(e)
    exit()

# =========================
# HANDLE MISSING VALUES
# =========================

try:
    missing = df.isnull().sum()

    total_missing = missing.sum()

    log(f"Total missing values: {total_missing}")

    # Missing values graph
    plt.figure(figsize=(10,5))

    missing.plot(kind='bar')

    plt.title("Missing Values Analysis")

    plt.xlabel("Features")

    plt.ylabel("Missing Values Count")

    plt.tight_layout()

    plt.savefig("outputs/missing_values.png")

    plt.close()

    log("Missing values figure generated")

    if total_missing > 0:
        df = df.dropna()

        log("Missing values removed")

except Exception as e:
    error(f"Missing value handling failed: {e}")
    exit()

# =========================
# CLASS BALANCING
# =========================

try:
    fraud = df[df[TARGET] == 1]

    normal = df[df[TARGET] == 0]

    log(f"Fraud transactions : {len(fraud)}")

    log(f"Normal transactions: {len(normal)}")

    if len(fraud) == 0 or len(normal) == 0:
        raise ValueError("One of the classes is empty")

    # Balance dataset
    normal_sample = normal.sample(
        n=len(fraud),
        random_state=42
    )

    df_balanced = pd.concat([
        fraud,
        normal_sample
    ])

    df_balanced = df_balanced.sample(
        frac=1,
        random_state=42
    )

    log("Class balancing completed")

    print("\nBalanced Dataset:")

    print(df_balanced[TARGET].value_counts())

    # Fraud vs Non-Fraud graph
    plt.figure(figsize=(6,5))

    df_balanced[TARGET].value_counts().plot(
        kind='bar'
    )

    plt.title("Fraud vs Non-Fraud Transactions")

    plt.xlabel("Class")

    plt.ylabel("Count")

    plt.tight_layout()

    plt.savefig(
        "outputs/fraud_vs_nonfraud.png"
    )

    plt.close()

    log("Fraud distribution graph generated")

except Exception as e:
    error(f"Class balancing failed: {e}")
    exit()

# =========================
# FEATURE / TARGET SPLIT
# =========================

try:
    X = df_balanced.drop(TARGET, axis=1)

    y = df_balanced[TARGET]

    if X.empty:
        raise ValueError("Feature dataset is empty")

    log("Feature-target split completed")

except Exception as e:
    error(f"Feature split failed: {e}")
    exit()

# =========================
# SAVE FEATURE NAMES
# =========================

try:
    joblib.dump(
        X.columns.tolist(),
        "models/feature_names.pkl"
    )

    log("Feature names saved")

except Exception as e:
    error(f"Feature name saving failed: {e}")

# =========================
# TRAIN TEST SPLIT
# =========================

try:
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    log("Train-test split completed")

    # Train-Test Split Figure
    train_size = len(X_train)

    test_size = len(X_test)

    labels = [
        "Training Data",
        "Testing Data"
    ]

    sizes = [
        train_size,
        test_size
    ]

    plt.figure(figsize=(6,6))

    plt.pie(
        sizes,
        labels=labels,
        autopct='%1.1f%%'
    )

    plt.title("Train-Test Data Split")

    plt.savefig(
        "outputs/train_test_split.png"
    )

    plt.close()

    log("Train-test split figure generated")

except Exception as e:
    error(f"Train-test split failed: {e}")
    exit()

# =========================
# FEATURE SCALING
# =========================

try:
    scaler = StandardScaler()

    before_scaling = X_train.iloc[:, 0].values[:100]

    X_train_scaled = scaler.fit_transform(X_train)

    X_test_scaled = scaler.transform(X_test)

    X_train_scaled = pd.DataFrame(
        X_train_scaled,
        columns=X.columns
    )

    X_test_scaled = pd.DataFrame(
        X_test_scaled,
        columns=X.columns
    )

    after_scaling = X_train_scaled.iloc[:, 0].values[:100]

    log("Feature scaling completed")

    # Feature Scaling Graph
    plt.figure(figsize=(10,5))

    plt.plot(
        before_scaling,
        label="Before Scaling"
    )

    plt.plot(
        after_scaling,
        label="After Scaling"
    )

    plt.title("Feature Scaling and Normalization")

    plt.xlabel("Samples")

    plt.ylabel("Values")

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        "outputs/feature_scaling.png"
    )

    plt.close()

    log("Feature scaling figure generated")

except Exception as e:
    error(f"Feature scaling failed: {e}")
    exit()

# =========================
# SAVE PROCESSED DATA
# =========================

try:
    X_train_scaled.to_csv(
        "data/processed/X_train.csv",
        index=False
    )

    X_test_scaled.to_csv(
        "data/processed/X_test.csv",
        index=False
    )

    y_train.to_csv(
        "data/processed/y_train.csv",
        index=False
    )

    y_test.to_csv(
        "data/processed/y_test.csv",
        index=False
    )

    log("Processed data saved successfully")

except Exception as e:
    error(f"Processed data saving failed: {e}")
    exit()

# =========================
# SAVE SCALER
# =========================

try:
    joblib.dump(
        scaler,
        "models/scaler.pkl"
    )

    log("Scaler saved successfully")

except Exception as e:
    error(f"Scaler saving failed: {e}")
    exit()

# =========================
# COMPLETED
# =========================

log("Preprocessing completed successfully")