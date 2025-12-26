# Load libraries

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, classification_report, roc_curve, precision_recall_curve, auc
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
import shap

# Step-1: Preprocessing Datasets
# Load datasets. Replace with EGR1 sensitive data to predict EGR1 sensitive features

kit_activated = pd.read_table("data/XGBoost/Kit_Activated_Training.txt")
kit_insensitive = pd.read_table("data/XGBoost/Kit_Insensitive_Training.txt")

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

# Train Neural Network Model

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
history = model.fit(X_train_scaled, y_train, epochs=35, batch_size=64, validation_data=(X_test_scaled, y_test), verbose=1)
y_prob_nn = model.predict(X_test_scaled).flatten()
y_pred_nn = (y_prob_nn > 0.5).astype(int)

# Evaluate AUROC & AUPR for Neural Networks

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

# Integrate the evaluation metrics for the trained classfication models

all_results = pd.DataFrame({
    "Model": ["Logistic Regression", "Random Forest", "XGBoost", "Neural Network"],
    "Accuracy": [log_results["Accuracy"], rf_results["Accuracy"], xgb_results["Accuracy"], nn_results["Accuracy"]],
    "AUROC": [log_results["AUROC"], rf_results["AUROC"], xgb_results["AUROC"], nn_results["AUROC"]],
    "AUPR": [log_results["AUPR"], rf_results["AUPR"], xgb_results["AUPR"], nn_results["AUPR"]]
})
#all_results.to_csv("Model_Comparison_for_Kit_Sensitive_data_table.csv",index=False)

# Plot Models Comparison

plt.figure(figsize=(8, 5))
sns.barplot(x="Model", y="Accuracy", data=all_results, palette="viridis")
plt.title("Model Accuracy Comparison", fontsize=14)
plt.ylabel("Accuracy")
plt.xticks(rotation=30)
plt.grid(axis="y", linestyle="", alpha=0.7)
#plt.savefig("Model_Comparison_for_Kit_Sensitive_data.pdf", format="pdf", bbox_inches="tight")
plt.show()

# Plot the AUROC and AUPR across the classification models trained

model_predictions = {
    "Logistic Regression": (y_prob_log, y_test),
    "Random Forest": (y_prob_rf, y_test),
    "XGBoost": (y_prob_xgb, y_test),
    "Neural Network": (y_prob_nn, y_test)
}
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
#plt.savefig("AUROC_Models_Comparison_Kit_Sensitive.pdf", format="pdf", bbox_inches="tight")
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
#plt.savefig("AUPR_Models_Comparison_Kit_Sensitive.pdf", format="pdf", bbox_inches="tight")
plt.show()

# Step-3: Proceed with XGBoost as the better classification model over others

from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, classification_report

# Initialize XGBoost with proper tuning

xgb_tuned = XGBClassifier(
    n_estimators=500,  
    learning_rate=0.1,  
    max_depth=10,  
    subsample=0.9,  
    colsample_bytree=0.9,
    reg_alpha=0.1,  # L1 regularization
    reg_lambda=0.5,  # L2 regularization
    random_state=42,
    n_jobs=-1
)
xgb_tuned.fit(X_train, y_train)
y_pred_xgb = xgb_tuned.predict(X_test)
y_prob_xgb = xgb_tuned.predict_proba(X_test)[:, 1]

# Predict on train data (to check overfitting)

y_pred_train_xgb = xgb_tuned.predict(X_train)
y_prob_train_xgb = xgb_tuned.predict_proba(X_train)[:, 1]
xgb_test_results = {
    "Accuracy": accuracy_score(y_test, y_pred_xgb),
    "AUROC": roc_auc_score(y_test, y_prob_xgb),
    "AUPR": average_precision_score(y_test, y_prob_xgb),
    "Classification Report": classification_report(y_test, y_pred_xgb)
}

# Training Set Evaluation (for Overfitting Detection)

xgb_train_results = {
    "Accuracy": accuracy_score(y_train, y_pred_train_xgb),
    "AUROC": roc_auc_score(y_train, y_prob_train_xgb),
    "AUPR": average_precision_score(y_train, y_prob_train_xgb)
}
def plot_auroc(y_true, y_prob, dataset_type, model_name="XGBoost"):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_score = roc_auc_score(y_true, y_prob)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"{model_name} (AUROC = {auc_score:.3f})", color="blue")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", alpha=0.7)  # Diagonal line
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.title(f"AUROC Curve - {model_name} ({dataset_type})")
    plt.legend()
    plt.grid()
    #plt.savefig(f"AUROC_XGBoost_new_{dataset_type}.pdf", format="pdf", bbox_inches="tight")
    plt.close()
    print(f"AUROC plot saved: AUROC_XGBoost_new_{dataset_type}.pdf")

plot_auroc(y_train, y_prob_train_xgb, "Train")
plot_auroc(y_test, y_prob_xgb, "Test")

# K-Fold Cross Validation

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_accuracy = cross_val_score(xgb_tuned, X_train, y_train, cv=kfold, scoring="accuracy")
cv_auroc = cross_val_score(xgb_tuned, X_train, y_train, cv=kfold, scoring="roc_auc")
cv_aupr = cross_val_score(xgb_tuned, X_train, y_train, cv=kfold, scoring="average_precision")

# Compute evaluation metrics

mean_acc, std_acc = cv_accuracy.mean(), cv_accuracy.std()
mean_auroc, std_auroc = cv_auroc.mean(), cv_auroc.std()
mean_aupr, std_aupr = cv_aupr.mean(), cv_aupr.std()

print("\nK-Fold Cross-Validation Results:")
print(f"Mean Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
print(f"Mean AUROC: {mean_auroc:.4f} ± {std_auroc:.4f}")
print(f"Mean AUPR: {mean_aupr:.4f} ± {std_aupr:.4f}")

# Train on full training set and evaluate test performance

xgb_tuned.fit(X_train, y_train)

# Predictions

y_pred_test = xgb_tuned.predict(X_test)
y_prob_test = xgb_tuned.predict_proba(X_test)[:, 1]

# Evaluate on test set

test_accuracy = accuracy_score(y_test, y_pred_test)
test_auroc = roc_auc_score(y_test, y_prob_test)
test_aupr = average_precision_score(y_test, y_prob_test)

print("\nXGBoost Test Set Results:")
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test AUROC: {test_auroc:.4f}")
print(f"Test AUPR: {test_aupr:.4f}")

# Test for overfitting

if abs(test_auroc - mean_auroc) > 0.02:
    print("AUROC Warning: Potential Overfitting Detected!")
if abs(test_aupr - mean_aupr) > 0.02:
    print("AUPR Warning: Potential Overfitting Detected!")
metrics = ["Accuracy", "AUROC", "AUPR"]
cv_means = [mean_acc, mean_auroc, mean_aupr]
cv_stds = [std_acc, std_auroc, std_aupr]
test_values = [test_accuracy, test_auroc, test_aupr]
plt.figure(figsize=(10, 6))
x = np.arange(len(metrics))
plt.bar(x - 0.2, cv_means, 0.4, yerr=cv_stds, capsize=0, label="Cross-Validation (Mean ± Std)", alpha=0.7)
plt.bar(x + 0.2, test_values, 0.4, label="Test Set", alpha=0.7)
plt.xticks(x, metrics, fontsize=12)
plt.ylabel("Score", fontsize=12)
plt.title("Cross-Validation vs Test Set Performance", fontsize=14)
plt.legend()
plt.ylim(0, 1)
#plt.savefig("Cross-Validation_vs_Test_Set_Performance.pdf", format="pdf", bbox_inches="tight")
plt.show()

# Step-4: Save the trained model

model_path = "data/KIT_xgb_tuned_model.json"
xgb_tuned.save_model(model_path)
print(f"XGBoost model saved at: {model_path}")

# Step-5: Evaluate the feature importances by SHAP

# Initialize SHAP

explainer = shap.TreeExplainer(xgb_tuned)
shap_values = explainer.shap_values(X_train)
shap_df = pd.DataFrame(shap_values, columns=X_train.columns)

# Compute mean SHAP values

mean_shap_values = shap_df.abs().mean(axis=0)

# Select features which are required in the model training and exclude the others

positive_shap_values = shap_df[shap_df > 0].mean()

predictors_df = positive_shap_values.sort_values(ascending=False).dropna()
#predictors_df.to_csv("Kit_Activation_TF_Importance_from_SHAP_XGBoost.csv")

# Step-6: Permutation testing to evaluate the significant features in helping in model performance. Use Step-1 again to read the datasets if running this seperately

# Load the model
import xgboost as xgb
from scipy.stats import norm # To calculate p-values for permutation testing

model = xgb.XGBClassifier()
model.load_model('data/XGBoost/KIT_xgb_tuned_model.json')
#model.classes_ = np.array([0, 1]) # Use if model fails to identify the class object

# Retrive the features of the trained model

expected_features = model.get_booster().feature_names
X_test_aligned = X_test[expected_features]

# Perform Permutation Importance

results = permutation_importance(
    model,
    X_test_aligned,
    y_test,
    scoring='roc_auc',
    n_repeats=100,
    random_state=42,
    n_jobs=None # Adjust jobs as required
)
importance_df = pd.DataFrame({
    'Feature': expected_features,
    'Importance_Mean': results.importances_mean,
    'Importance_Std': results.importances_std
})
importance_df = importance_df.sort_values(by='Importance_Mean', ascending=False)
importance_df['z_score'] = importance_df['Importance_Mean'] / (importance_df['Importance_Std'] + 1e-9)
importance_df['p_value'] = 1 - norm.cdf(importance_df['z_score'])
#importance_df.to_csv("Kit_permutation_importance_table_100_permutations.csv", index=False)

# Accuracy and AUROC for the tuning adjusted XGBoost Model ('xgb_tuned')

y_pred = model.predict(X_test)
y_probs = model.predict_proba(X_test)[:, 1]
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# Calculate AUROC

fpr, tpr, thresholds = roc_curve(y_test, y_probs)
roc_auc = auc(fpr, tpr)

# Plot

plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, color='#2c3e50', lw=2.5, label=f'Model ROC (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='grey', lw=1, linestyle='--') # Random guess line
plt.xlim([-0.02, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('XGBoost AUROC Curve', fontsize=14, fontweight='bold')
plt.legend(loc="lower right", frameon=True)
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
#plt.savefig("xgboost-tuned-auroc-curve.pdf", dpi=300)
plt.show()

