from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

# Resolve paths (same as 01)
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
data_path = project_root / "data" / "clean_ecommerce_data.csv"

# Load cleaned dataset
df = pd.read_csv(data_path)

# Separate features (X) and target (y)
X = df.drop(columns=["Churned"])
y = df["Churned"]

# Split into 80% training and 20% testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# Scale numeric features for Logistic Regression Model
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 1. Train Logistic Regression
log_reg = LogisticRegression(max_iter=1000, random_state=42)
log_reg.fit(X_train_scaled, y_train)

log_pred = log_reg.predict(X_test_scaled)
log_proba = log_reg.predict_proba(X_test_scaled)[:, 1]

# 2. Train Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)
rf_proba = rf_model.predict_proba(X_test)[:, 1]

# Evaluate Models
print("=" * 50)
print("LOGISTIC REGRESSION PERFORMANCE")
print("=" * 50)
print(classification_report(y_test, log_pred, digits=3))
print(f"ROC-AUC Score: {roc_auc_score(y_test, log_proba):.3f}\n")

print("=" * 50)
print("RANDOM FOREST PERFORMANCE")
print("=" * 50)
print(classification_report(y_test, rf_pred, digits=3))
print(f"ROC-AUC Score: {roc_auc_score(y_test, rf_proba):.3f}\n")

# Extract Random Forest Feature Importances
importances = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=False)
print("Top 10 Most Important Features (Random Forest):")
print(importances.head(10).round(4))