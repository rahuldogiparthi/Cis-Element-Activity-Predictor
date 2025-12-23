import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# Load datasets
kit_activated = pd.read_table("data/Kit_Activated_Training.txt")
kit_insensitive = pd.read_table("data/Kit_Insensitive_Training.txt")

# Add labels: 1 for Kit Activated, 0 for Kit Insensitive
kit_activated["Label"] = 1
kit_insensitive["Label"] = 0

# Combine datasets
df = pd.concat([kit_activated, kit_insensitive], ignore_index=True)

# Drop non-binary columns
non_binary_cols = ["chr", "start", "end", "Genes", "ABC-Type", "TF", "CellType"]
df = df.drop(columns=non_binary_cols, errors='ignore')

# Shuffle dataset
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Split features and labels
X = df.drop(columns=["Label"])
y = df["Label"]

# Handle class imbalance using SMOTE
smote = SMOTE(sampling_strategy="auto", random_state=42, k_neighbors=5)
X_balanced, y_balanced = smote.fit_resample(X, y)

# Split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.2, stratify=y_balanced, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# print(f"Data Loaded & Preprocessed. Train shape: {X_train.shape}, Test shape: {X_test.shape}")
