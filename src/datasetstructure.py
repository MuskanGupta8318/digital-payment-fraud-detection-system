import pandas as pd
import matplotlib.pyplot as plt
import os

# =========================
# CREATE OUTPUT FOLDER
# =========================

os.makedirs("outputs", exist_ok=True)

# =========================
# LOAD DATASET
# =========================

file_path = "data/creditcard.csv"

try:
    df = pd.read_csv(file_path)

    print("Dataset loaded successfully")

except Exception as e:

    print(f"Error loading dataset: {e}")

    exit()

# =====================================================
# 1. DATASET LOADING AND STRUCTURE OUTPUT
# =====================================================

try:
    dataset_info = {
        "Property": [
            "Number of Rows",
            "Number of Columns",
            "Target Column"
        ],

        "Value": [
            df.shape[0],
            df.shape[1],
            "Class"
        ]
    }

    info_df = pd.DataFrame(dataset_info)

    fig, ax = plt.subplots(
        figsize=(8,3)
    )

    ax.axis('off')

    table = ax.table(
        cellText=info_df.values,
        colLabels=info_df.columns,
        cellLoc='center',
        loc='center'
    )

    table.auto_set_font_size(False)

    table.set_fontsize(12)

    table.scale(1.2, 1.8)

    plt.title(
        "Dataset Loading and Structure Output",
        pad=20
    )

    plt.savefig(
        "outputs/dataset_structure_output.png",
        bbox_inches='tight'
    )

    plt.close()

    print(
        "Dataset structure output generated"
    )

except Exception as e:

    print(
        f"Dataset structure figure failed: {e}"
    )

# =====================================================
# 2. DATASET STRUCTURE / COLUMN REPRESENTATION
# =====================================================

try:
    columns_data = pd.DataFrame({

        "Feature Name": df.columns,

        "Data Type": df.dtypes.astype(str)
    })

    fig, ax = plt.subplots(
        figsize=(10,12)
    )

    ax.axis('off')

    table = ax.table(
        cellText=columns_data.values,
        colLabels=columns_data.columns,
        cellLoc='center',
        loc='center'
    )

    table.auto_set_font_size(False)

    table.set_fontsize(10)

    table.scale(1.2, 1.5)

    plt.title(
        "Dataset Structure and Feature Representation",
        pad=20
    )

    plt.savefig(
        "outputs/columns_representation.png",
        bbox_inches='tight'
    )

    plt.close()

    print(
        "Columns representation figure generated"
    )

except Exception as e:

    print(
        f"Columns representation failed: {e}"
    )

# =====================================================
# COMPLETED
# =====================================================

print(
    "\\nDataset structure figures generated successfully"
)