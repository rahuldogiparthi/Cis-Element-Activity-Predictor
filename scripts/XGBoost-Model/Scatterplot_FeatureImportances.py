# Load libraries

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load data

df = pd.read_csv("data/Kit_Activation_TF_Importance_from_SHAP_XGBoost.csv", header=None, names=["feature", "importance"])
df["importance"] = pd.to_numeric(df["importance"].astype(str).str.replace(",", ""), errors="coerce")
df = df.dropna().sort_values("importance", ascending=False).reset_index(drop=True)
sns.set_theme(style="whitegrid", context="talk")
plt.figure(figsize=(7, 4))

# Scatter plot

plt.scatter(range(1, len(df) + 1), df["importance"], s=25, color="#4C72B0")

# Labels

plt.xlabel("Feature rank", fontsize=12)
plt.ylabel("Importance", fontsize=12)
plt.title("Feature Importance Distribution", loc="left", weight="bold", fontsize=14)
sns.despine()
plt.tight_layout()
plt.savefig("Kit Sensitivity Feature_Importance_Distribution.pdf", format="pdf", bbox_inches="tight")
plt.show()
