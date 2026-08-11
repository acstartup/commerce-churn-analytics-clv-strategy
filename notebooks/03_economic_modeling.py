from pathlib import Path
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Resolve paths (same as 01 and 02)
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
data_path = project_root / "data" / "clean_ecommerce_data.csv"

# Load cleaned dataset
df = pd.read_csv(data_path)

X = df.drop(columns=["Churned"])
y = df["Churned"]

# Train model to generate churn probabilities
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Evaluate high-risk, high-value segment
eval_df = X_test.copy()
eval_df["Churn_Probability"] = rf_model.predict_proba(X_test)[:, 1]

median_ltv = eval_df["Lifetime_Value"].median()
at_risk_mask = (eval_df["Churn_Probability"] >= 0.50) & (eval_df["Lifetime_Value"] >= median_ltv)
target_segment = eval_df[at_risk_mask]

total_target_customers = len(target_segment)
total_at_risk_ltv = target_segment["Lifetime_Value"].sum()

# Define sensitivity scenarios, assumptions of discount to lift, low discount = too small for customers to recognize
scenarios = [
    {"name": "Conservative (5%)", "discount": 0.05, "lift": 0.035},
    {"name": "Base Case (10%)", "discount": 0.10, "lift": 0.150},
    {"name": "Aggressive (20%)", "discount": 0.20, "lift": 0.225},
]

results = []
for sc in scenarios:
    gross_retained = total_at_risk_ltv * sc["lift"]
    campaign_cost = total_at_risk_ltv * sc["discount"]
    net_protected = gross_retained - campaign_cost
    roi = (net_protected / campaign_cost) * 100 if campaign_cost > 0 else 0

    results.append({
        "Scenario": sc["name"],
        "Discount Rate": f"{sc['discount']*100:.0f}%",
        "Retention Lift": f"{sc['lift']*100:.1f}%",
        "Gross Retained": f"${gross_retained:,.2f}",
        "Campaign Cost": f"${campaign_cost:,.2f}",
        "Net Protected": f"${net_protected:,.2f}",
        "ROI (%)": f"{roi:.1f}%"
    })

# Output Summary
print("=" * 65)
print("SENSITIVITY ANALYSIS: TARGETED DISCOUNT SCENARIOS")
print("=" * 65)
print(f"Target Segment: {total_target_customers:,} High-Value At-Risk Customers")
print(f"Total Revenue Exposure (LTV): ${total_at_risk_ltv:,.2f}\n")

results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))
print("=" * 65)