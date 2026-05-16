import matplotlib.pyplot as plt
import os

# Create outputs folder
os.makedirs("outputs", exist_ok=True)

try:
    # Example transaction risk scores
    transactions = [
        "T1", "T2", "T3", "T4", "T5",
        "T6", "T7", "T8", "T9", "T10"
    ]

    risk_scores = [
        0.10, 0.25, 0.45, 0.62, 0.80,
        0.15, 0.92, 0.55, 0.30, 0.72
    ]

    # Create graph
    plt.figure(figsize=(10,5))

    bars = plt.bar(transactions, risk_scores)

    # Risk threshold lines
    plt.axhline(
        y=0.5,
        color='blue',
        linestyle='--',
        label='Medium Risk Threshold'
    )

    plt.axhline(
        y=0.8,
        color='Red',
        linestyle='--',
        label='High Risk Threshold'
    )

    # Labels
    plt.xlabel("Transactions")
    plt.ylabel("Risk Score")

    plt.title("Transaction Risk Score Analysis")

    plt.ylim(0, 1)

    plt.legend()

    # Add values above bars
    for bar in bars:

        yval = bar.get_height()

        plt.text(
            bar.get_x() + bar.get_width()/2,
            yval + 0.02,
            f"{yval:.2f}",
            ha='center'
        )

    # Save figure
    plt.savefig(
        "outputs/risk_score_analysis.png"
    )

    print("[INFO] Risk score graph generated successfully")

    plt.show()

except Exception as e:
    print(f"[ERROR] Risk score graph failed: {e}")