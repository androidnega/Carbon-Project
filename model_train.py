import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

# Step 1: Load and explore the dataset
print("=== LOADING DATASET ===")
df = pd.read_csv("data/Carbon Emission.csv")

# Display the top 5 rows
print("\n=== TOP 5 ROWS ===")
print(df.head())

# Check the columns and missing values
print("\n=== DATASET INFO ===")
print(df.info())
print("\n=== MISSING VALUES ===")
print(df.isnull().sum())

print("\n=== DATASET SHAPE ===")
print(f"Dataset shape: {df.shape}")

print("\n=== ALL COLUMNS ===")
print(df.columns.tolist())

# Step 2: Data cleaning and preparation
print("\n=== DATA CLEANING AND PREPARATION ===")

# Check if we have the expected columns (adapting to actual dataset structure)
# The dataset has CarbonEmission as target and various lifestyle features
# We'll need to map relevant features to the expected ones or use available ones

# Drop rows with missing values
print(f"Rows before cleaning: {len(df)}")
df_clean = df.dropna()
print(f"Rows after cleaning: {len(df_clean)}")

# For this dataset, we'll use relevant features that could relate to:
# - product_weight: Monthly Grocery Bill (as proxy for consumption)
# - shipping_distance: Vehicle Monthly Distance Km 
# - packaging_type: We'll create this from available categorical data

# Create feature mappings based on available data
print("\n=== FEATURE ENGINEERING ===")

# Use Vehicle Monthly Distance as shipping_distance equivalent
df_clean['shipping_distance'] = df_clean['Vehicle Monthly Distance Km']

# Use Monthly Grocery Bill as product_weight equivalent  
df_clean['product_weight'] = df_clean['Monthly Grocery Bill']

# Create packaging_type from Transport method
df_clean['packaging_type'] = df_clean['Transport']

# Encode categorical variable (packaging_type)
le = LabelEncoder()
df_clean['packaging_type_encoded'] = le.fit_transform(df_clean['packaging_type'])

print("Packaging types:", df_clean['packaging_type'].unique())
print("Encoded values:", df_clean['packaging_type_encoded'].unique())

# Step 3: Select features and target
# Features: product_weight, shipping_distance, packaging_type_encoded
# Target: CarbonEmission
X = df_clean[['product_weight', 'shipping_distance', 'packaging_type_encoded']]
y = df_clean['CarbonEmission']

print(f"\nFeatures shape: {X.shape}")
print(f"Target shape: {y.shape}")

print("\n=== FEATURE STATISTICS ===")
print(X.describe())

print("\n=== TARGET STATISTICS ===")
print(y.describe())
