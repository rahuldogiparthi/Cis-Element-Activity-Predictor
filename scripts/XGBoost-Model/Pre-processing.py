import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, classification_report, roc_curve, precision_recall_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from xgboost import XGBClassifier
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import seaborn as sns

# Step-1: Preprocessing Datasets
# Load datasets
kit_activated = pd.read_table("data/Kit_Activated_Training.txt")
kit_insensitive = pd.read_table("data/Kit_Insensitive_Training.txt")

# Add labels: 1 for Kit Activated, 0 for Kit Insensitive
kit_activated["Label"] = 1
kit_insensitive["Label"] = 0

df = pd.concat([kit_activated, kit_insensitive], ignore_index=True)

# Drop non-binary columns
non_binary_cols = ["chr", "start", "end", "Genes", "ABC-Type", "TF", "CellType"]
df = df.drop(columns=non_binary_cols, errors='ignore')

# Shuffle dataset
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Split features and labels
X = df.drop(columns=["Label"])
y = df["Label"]

# Handle class imbalances using SMOTE
smote = SMOTE(sampling_strategy="auto", random_state=42, k_neighbors=5)
X_balanced, y_balanced = smote.fit_resample(X, y)

# Train-test splits
X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.2, stratify=y_balanced, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step-2: Evaluate the classification models performance after training the dataset
# Train Logistic Regression
log_reg = LogisticRegression(max_iter=500, n_jobs=-1)
log_reg.fit(X_train_scaled, y_train)

y_pred_log = log_reg.predict(X_test_scaled)
y_prob_log = log_reg.predict_proba(X_test_scaled)[:, 1]

log_results = {
    "Accuracy": accuracy_score(y_test, y_pred_log),
    "AUROC": roc_auc_score(y_test, y_prob_log),
    "AUPR": average_precision_score(y_test, y_prob_log),
    "Classification Report": classification_report(y_test, y_pred_log)
}
print("Logistic Regression Results:\n", log_results)

# Train Random Forest
rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)
y_prob_rf = rf.predict_proba(X_test)[:, 1]

rf_results = {
    "Accuracy": accuracy_score(y_test, y_pred_rf),
    "AUROC": roc_auc_score(y_test, y_prob_rf),
    "AUPR": average_precision_score(y_test, y_prob_rf),
    "Classification Report": classification_report(y_test, y_pred_rf)
}
print("Random Forest Results:\n", rf_results)

# Train XGBoost
xgb = XGBClassifier(n_estimators=500, learning_rate=0.05, max_depth=8, subsample=0.8, colsample_bytree=0.8, 
                    random_state=42, n_jobs=-1)
xgb.fit(X_train, y_train)

y_pred_xgb = xgb.predict(X_test)
y_prob_xgb = xgb.predict_proba(X_test)[:, 1]

xgb_results = {
    "Accuracy": accuracy_score(y_test, y_pred_xgb),
    "AUROC": roc_auc_score(y_test, y_prob_xgb),
    "AUPR": average_precision_score(y_test, y_prob_xgb),
    "Classification Report": classification_report(y_test, y_pred_xgb)
}
print("XGBoost Results: \n", xgb_results)

# Build Neural Network Model
model = Sequential([
    Dense(128, activation="relu", input_shape=(X_train_scaled.shape[1],)),
    BatchNormalization(),
    Dropout(0.3),
    Dense(64, activation="relu"),
    BatchNormalization(),
    Dropout(0.2),
    Dense(32, activation="relu"),
    Dense(1, activation="sigmoid")  # Binary classification
])

model.compile(optimizer=Adam(learning_rate=0.0005), loss="binary_crossentropy", metrics=["accuracy"])

# Train Model
history = model.fit(X_train_scaled, y_train, epochs=35, batch_size=64, validation_data=(X_test_scaled, y_test), verbose=1)

y_prob_nn = model.predict(X_test_scaled).flatten()
y_pred_nn = (y_prob_nn > 0.5).astype(int)

# Evaluate AUROC & AUPR
nn_auroc = roc_auc_score(y_test, y_prob_nn)
nn_aupr = average_precision_score(y_test, y_prob_nn)
nn_accuracy = sum(y_pred_nn == y_test) / len(y_test)

nn_results = {
    "Accuracy": nn_accuracy,
    "AUROC": nn_auroc,
    "AUPR": nn_aupr
}

print(f"Neural Network Test Accuracy: {nn_accuracy:.4f}")
print(f"Neural Network AUROC: {nn_auroc:.4f}")
print(f"Neural Network AUPR: {nn_aupr:.4f}")

all_results = pd.DataFrame({
    "Model": ["Logistic Regression", "Random Forest", "XGBoost", "Neural Network"],
    "Accuracy": [log_results["Accuracy"], rf_results["Accuracy"], xgb_results["Accuracy"], nn_results["Accuracy"]],
    "AUROC": [log_results["AUROC"], rf_results["AUROC"], xgb_results["AUROC"], nn_results["AUROC"]],
    "AUPR": [log_results["AUPR"], rf_results["AUPR"], xgb_results["AUPR"], nn_results["AUPR"]]
})

# Plot Models Comparison
plt.figure(figsize=(8, 5))
sns.barplot(x="Model", y="Accuracy", data=all_results, palette="viridis")
plt.title("Model Accuracy Comparison", fontsize=14)
plt.ylabel("Accuracy")
plt.xticks(rotation=30)
plt.grid(axis="y", linestyle="", alpha=0.7)
#plt.savefig("Model_Comparison.pdf", format="pdf", bbox_inches="tight")
plt.show()

# Plot the AUROC and AUPR across the classification models trained
model_predictions = {
    "Logistic Regression": (y_prob_log, y_test),
    "Random Forest": (y_prob_rf, y_test),
    "XGBoost": (y_prob_xgb, y_test),
    "Neural Network": (y_prob_nn, y_test)
}

# Define Colors for Plot
colors = ["blue", "green", "red", "purple"]

# Plot AUROC Curves
plt.figure(figsize=(8, 6))
for (model, (prob, true_labels)), color in zip(model_predictions.items(), colors):
    fpr, tpr, _ = roc_curve(true_labels, prob)
    plt.plot(fpr, tpr, label=f"{model} (AUROC = {roc_auc_score(true_labels, prob):.3f})", color=color)

plt.plot([0, 1], [0, 1], linestyle="--", color="gray", alpha=0.7)  # Diagonal line
plt.xlabel("False Positive Rate (FPR)")
plt.ylabel("True Positive Rate (TPR)")
plt.title("AUROC Curves for All Models")
plt.legend()
plt.grid()
#plt.savefig("AUROC_Models_Comparison.pdf", format="pdf", bbox_inches="tight")
plt.show()

# Plot AUPR Curves
plt.figure(figsize=(8, 6))
for (model, (prob, true_labels)), color in zip(model_predictions.items(), colors):
    precision, recall, _ = precision_recall_curve(true_labels, prob)
    plt.plot(recall, precision, label=f"{model} (AUPR = {average_precision_score(true_labels, prob):.3f})", color=color)

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("AUPR Curves for All Models")
plt.legend()
plt.grid()
#plt.savefig("AUPR_Models_Comparison.pdf", format="pdf", bbox_inches="tight")
plt.show()

# Step-3: Proceed with XGBoost as the better classification model over others


