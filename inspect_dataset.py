import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_csv('data/Carbon Emission.csv')

print("=== DATASET INSPECTION ===")
print(f"Dataset shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print("\n=== FIRST 5 ROWS ===")
print(df.head())
print("\n=== DATA TYPES ===")
print(df.dtypes)
print("\n=== BASIC STATISTICS ===")
print(df.describe())
print("\n=== MISSING VALUES ===")
print(df.isnull().sum())
print("\n=== UNIQUE VALUES PER COLUMN ===")
for col in df.columns:
    print(f"{col}: {df[col].nunique()} unique values")
