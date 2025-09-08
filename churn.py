# ===============================
# 1. Import Libraries
# ===============================
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# ===============================
# 2. Load Dataset
# ===============================
df = pd.read_csv("C:/Users/saksh/OneDrive - UPES/Project/WA_Fn-UseC_-Telco-Customer-Churn.csv")

print("Shape of dataset:", df.shape)
print(df.head())

# ===============================
# 3. Basic Cleaning
# ===============================
# Drop CustomerID (not useful for prediction)
df.drop(['customerID'], axis=1, inplace=True)

# Fix TotalCharges column (it has blank strings, convert to numeric)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Handle missing values (after conversion some NaN will appear)
df.dropna(inplace=True)

# Encode categorical variables
for col in df.select_dtypes(include=['object']).columns:
    df[col] = LabelEncoder().fit_transform(df[col])

print("\nDataset after cleaning and encoding:")
print(df.head())

# ===============================
# 4. Split Features & Target
# ===============================
X = df.drop('Churn', axis=1)
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale numerical features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ===============================
# 5. Train Model (Random Forest)
# ===============================
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\n🔎 Model Evaluation")
print(classification_report(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_pred))

# ===============================
# 6. Exploratory Data Analysis (EDA)
# ===============================
plt.figure(figsize=(6,4))
sns.countplot(x='Churn', data=df)
plt.title("Churn Distribution")
plt.show()

plt.figure(figsize=(8,5))
sns.boxplot(x='Churn', y='MonthlyCharges', data=df)
plt.title("Monthly Charges vs Churn")
plt.show()

plt.figure(figsize=(8,5))
sns.boxplot(x='Churn', y='tenure', data=df)
plt.title("Tenure vs Churn")
plt.show()

plt.figure(figsize=(12,8))
sns.heatmap(df.corr(), cmap="coolwarm", annot=False)
plt.title("Correlation Heatmap")
plt.show()

# ===============================
# 7. Feature Importance
# ===============================
importances = model.feature_importances_
features = X.columns
feat_importance = pd.Series(importances, index=features).sort_values(ascending=False)

plt.figure(figsize=(10,6))
sns.barplot(x=feat_importance[:10], y=feat_importance.index[:10])
plt.title("Top 10 Features Driving Churn")
plt.show()

# ===============================
# 8. Save Trained Model
# ===============================
import pickle

with open("churn_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved as churn_model.pkl")
