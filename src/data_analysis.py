import pandas as pd

# Load dataset
df = pd.read_csv("data/creditcard.csv")

# Basic info
print(" Shape of dataset:", df.shape)

print("\n Columns:")
print(df.columns)

print("\n Data Types:")
print(df.dtypes)

print("\n Missing Values:")
print(df.isnull().sum())

print("\n First 5 Rows:")
print(df.head())

print("\n Statistical Summary:")
print(df.describe())

# Fraud vs Normal distribution
print("\n Class Distribution:")
print(df['Class'].value_counts())