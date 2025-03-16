import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score

# Load dataset
dataset_path = 'C:/Users/Ncjy1/OneDrive/Desktop/Datasets/Churn/new_synthetic_8.csv'
synthetic_data = pd.read_csv(dataset_path)

# Drop irrelevant columns
synthetic_data.drop(['customerID', 'Email'], axis=1, inplace=True)

# Convert categorical binary columns to 0/1
binary_columns = ['Partner', 'Dependents', 'PhoneService', 'MultipleLines', 
                  'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                  'TechSupport', 'StreamingTV', 'StreamingMovies', 
                  'PaperlessBilling', 'PaymentTimeliness', 'gender', 'SatisfactionScore']

synthetic_data[binary_columns] = synthetic_data[binary_columns].replace({
    'Yes': 1, 'No': 0, 'No phone service': 0, 'No internet service': 0, 
    'On-time': 1, 'Late': 0, 'Male': 1, 'Female': 0, 'High': 1, 'Low': 0
})

# One-hot encoding for categorical variables
df_encoded = pd.get_dummies(synthetic_data, drop_first=True)

# Define features and target
X = df_encoded.drop('Churn', axis=1)
y = df_encoded['Churn']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Save feature columns
joblib.dump(X.columns.tolist(), 'X_train_columns.pkl')

# Train logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# Model evaluation
#print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
###print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Feature importance
#coefficients = pd.DataFrame({'Feature': X.columns, 'Coefficient': model.coef_[0]})
#print("Feature Importance:\n", coefficients.sort_values(by='Coefficient', ascending=False))

# Save model
joblib.dump(model, 'churn_model.pkl')

# Load trained model
model = joblib.load('churn_model.pkl')

# Load new dataset
input_path = 'C:/Users/Ncjy1/OneDrive/Desktop/Datasets/Churn/new_synthetic_7.csv'
input_dataset = pd.read_csv(input_path)

# Preprocessing
input_dataset[binary_columns] = input_dataset[binary_columns].replace({
    'Yes': 1, 'No': 0, 'No phone service': 0, 'No internet service': 0, 
    'On-time': 1, 'Late': 0, 'Male': 1, 'Female': 0, 'High': 1, 'Low': 0
})

# One-hot encoding
input_dataset_encoded = pd.get_dummies(input_dataset, drop_first=True)

# Load feature columns
X_train_columns = joblib.load('X_train_columns.pkl')

# Handle missing columns
missing_cols = set(X_train_columns) - set(input_dataset_encoded.columns)
for col in missing_cols:
    input_dataset_encoded[col] = 0

# Log extra columns (for debugging)
extra_cols = set(input_dataset_encoded.columns) - set(X_train_columns)
if extra_cols:
    print("Warning: Extra columns found in new dataset -", extra_cols)

# Ensure correct column order
input_dataset_encoded = input_dataset_encoded[X_train_columns]

# Predict churn
input_dataset['Churn_Prediction'] = model.predict(input_dataset_encoded)
input_dataset['Churn_Probability'] = model.predict_proba(input_dataset_encoded)[:, 1]

# Save predictions
output_path = 'C:/Users/Ncjy1/OneDrive/Desktop/Datasets/Churn/new_synthetic_7_churned.csv'
input_dataset.to_csv(output_path, index=False)

print(f"Predictions saved successfully to '{output_path}'")
