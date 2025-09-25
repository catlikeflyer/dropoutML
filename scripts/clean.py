import pandas as pd
from sklearn.model_selection import train_test_split
import os
import joblib

print("Starting data cleaning and preprocessing...")

# Load your specific dataset
df = pd.read_csv('data/student_data.csv')

# 1. Define Target and Features
# Drop 'Final_Grade' to prevent data leakage
features_to_drop = ['Final_Grade']
df_features = df.drop(columns=features_to_drop)

# Convert the target variable 'Dropped_Out' from True/False to 1/0
df_features['Dropped_Out'] = df_features['Dropped_Out'].astype(int)
y = df_features.pop('Dropped_Out')
X = df_features

# 2. Handle Binary Features (Label Encoding)
binary_cols = [col for col in X.columns if X[col].dropna().nunique() == 2]
for col in binary_cols:
    # Use pd.factorize to safely convert to 0 and 1
    X[col] = pd.factorize(X[col])[0]

# 3. Handle Multi-Categorical Features (One-Hot Encoding)
categorical_cols = X.select_dtypes(include=['object']).columns
X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

# 4. Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 5. Save Processed Data
processed_dir = 'data/processed'
os.makedirs(processed_dir, exist_ok=True)
X_train.to_csv(f'{processed_dir}/X_train.csv', index=False)
X_test.to_csv(f'{processed_dir}/X_test.csv', index=False)
y_train.to_csv(f'{processed_dir}/y_train.csv', index=False)
y_test.to_csv(f'{processed_dir}/y_test.csv', index=False)

# Save the training columns for the API
joblib.dump(X_train.columns, 'src/training_columns.pkl')

print("Data processing complete. Files saved in data/processed/.")
print(f"Training features count: {len(X_train.columns)}")