from pathlib import Path
import pandas as pd
import numpy as np

# Resolve paths (01,02,03)
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
data_path = project_root / "data" / "clean_ecommerce_data.csv"

df = pd.read_csv(data_path)

# 1. Scenario vs ROI Data
scenarios_data = [
    {"Scenario": "Conservative (5%)", "Discount_Rate": 0.05, "Retention_Lift": 0.035, "Gross_Retained": 35029.72, "Campaign_Cost": 50042.46, "Net_Protected_Revenue": -15012.74, "ROI_Percent": -30.0},
    {"Scenario": "Base Case (10%)", "Discount_Rate": 0.10, "Retention_Lift": 0.150, "Gross_Retained": 150127.39, "Campaign_Cost": 100084.93, "Net_Protected_Revenue": 50042.46, "ROI_Percent": 50.0},
    {"Scenario": "Aggressive (20%)", "Discount_Rate": 0.20, "Retention_Lift": 0.225, "Gross_Retained": 225191.09, "Campaign_Cost": 200169.85, "Net_Protected_Revenue": 25021.23, "ROI_Percent": 12.5},
]
df_scenarios = pd.DataFrame(scenarios_data)
df_scenarios.to_csv(project_root / "data" / "tableau_scenarios.csv", index=False)

# 2. Correlation Data
num_cols = df.select_dtypes(include=["float64", "int64"]).columns.drop("Churned")
correlations = df[num_cols].corrwith(df["Churned"]).round(4)

df_corr = pd.DataFrame({
    "Feature": correlations.index,
    "Correlation_With_Churn": correlations.values,
    "Impact_Type": np.where(correlations.values > 0, "Churn Driver (+)", "Retention Driver (-)")
}).sort_values(by="Correlation_With_Churn", ascending=False)

df_corr.to_csv(project_root / "data" / "tableau_correlations.csv", index=False)

# 3. Friction & Feature Importance Data
# Aggregate Churn Rate by Customer Service Calls and Cart Abandonment Bins
df["Cart_Abandonment_Bin"] = pd.qcut(df["Cart_Abandonment_Rate"], q=5, labels=["Very Low", "Low", "Medium", "High", "Very High"])
df_friction = df.groupby(["Customer_Service_Calls", "Cart_Abandonment_Bin"], observed=False).agg(
    Total_Customers=("Churned", "count"),
    Churned_Customers=("Churned", "sum"),
    Avg_LTV=("Lifetime_Value", "mean")
).reset_index()

df_friction["Churn_Rate"] = (df_friction["Churned_Customers"] / df_friction["Total_Customers"]).round(4)
df_friction.to_csv(project_root / "data" / "tableau_friction_segments.csv", index=False)
print("\nSaved clean dataset")