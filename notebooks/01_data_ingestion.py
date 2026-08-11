from pathlib import Path
import pandas as pd

# Resolve paths relative to this script location
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
data_path = project_root / "data" / "ecommerce_dataset.csv"

# Load ecommerce dataset
df = pd.read_csv(data_path)

print(f"Loaded raw dataset with shape: {df.shape}")

# Drop any rows containing missing values
df = df.dropna().reset_index(drop=True)
print(f"Dataset shape after dropping nulls: {df.shape}")

print("\nTarget ('Churned') distribution:")
print(df["Churned"].value_counts(normalize=True).round(3))

# Calculate correlations with col x churn
num_cols = df.select_dtypes(include=["float64", "int64"]).columns.drop("Churned")
correlations = df[num_cols].corrwith(df["Churned"]).sort_values(ascending=False)

print("\nTop positive correlations with Churned (correlates with leaving):")
print(correlations.head(5).round(3))

print("\nTop negative correlations with Churned (correlates with retention):")
print(correlations.tail(5).round(3))

# Drop high-cardinality location field and convert remaining text columns to binary flags
df_model = df.drop(columns=["City"])
cat_cols = ["Gender", "Country", "Signup_Quarter"]
df_model = pd.get_dummies(df_model, columns=cat_cols, drop_first=True)

# Save processed dataset into cleaned folder for model training
output_path = project_root / "data" / "clean_ecommerce_data.csv"
df_model.to_csv(output_path, index=False)
print(f"\nSaved clean dataset to {output_path}")