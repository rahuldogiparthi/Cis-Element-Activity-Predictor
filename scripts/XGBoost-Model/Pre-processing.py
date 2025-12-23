import pandas as pd
import numpy as np


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

