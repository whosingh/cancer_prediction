# import os
# print(os.getcwd())

# ===== Step 1: Import Libraries ===========================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve, auc, fbeta_score, log_loss

# ===== Step 2: Load and Explore the Dataset =================

# Load the dataset
df = pd.read_csv('breast_cancer_data.csv')
# print(df.head) # Explore the Data 
# print(df.info()) # Check for missing values and data types
# print(df.describe()) # Statistical summary of the dataset

# ===== Step 3: Data Preprocessing ===========================

# Check for missing values
# print(df.isnull().sum())

# Check for duplicate rows
# print(df.duplicated().sum())

# Split the data into features (X) and target (y)
X = df.drop('target', axis=1)
type(X)

y = df['target']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features (scaling)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# print(X_test)
# print(df['target'].value_counts())

# ===== Step 4: EDA and Visualization =======================

# Get value counts (number of each class)
target_counts = df['target'].value_counts()

# Create pie chart
plt.figure(figsize=(4, 4))
plt.pie(
    target_counts,
    labels=target_counts.index,        # label as 0 and 1
    autopct='%1.1f%%',                 # show % with 1 decimal
    startangle=90,                     # rotate chart for better alignment
    colors=['skyblue', 'lightcoral'],  # optional custom colors
    explode=(0.05, 0.05)               # slightly separate slices for emphasis
)

# Add title
plt.title("Distribution of Target Variable (%)")

# Display the pie chart
# plt.show()

# Compute correlation matrix
correlation_matrix = df.corr(numeric_only=True)

# Extract correlation values with the target variable
target_corr = correlation_matrix['target'].sort_values(ascending=False)

# Plot the correlation as a bar chart
plt.figure(figsize=(10, 6))
sns.barplot(
    x=target_corr.values,
    y=target_corr.index,
    palette='coolwarm'
)

# Add title and labels
plt.title('Correlation of Each Feature with Target', fontsize=14)
plt.xlabel('Correlation Coefficient', fontsize=12)
plt.ylabel('Features', fontsize=12)

# Add grid for better readability
plt.grid(axis='x', linestyle='--', alpha=0.7)

# Show the plot
# plt.show()

# Visualize a correlation heatmap:
correlation_matrix = df.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm')
plt.title('Correlation Heatmap')
# plt.show()

# ===== Step 5: Model Training and Evaluation ==========

# Train a logistic regression model
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
# print(f'Accuracy: {accuracy:.2f}')

# Confusion Matrix
confusion = confusion_matrix(y_test, y_pred)
print('Confusion Matrix:')
# print(confusion)

# Classification Report
classification_rep = classification_report(y_test, y_pred)
print('Classification Report:')
# print(classification_rep)

# ✅ F-beta Scores
f2 = fbeta_score(y_test, y_pred, beta=2)
f05 = fbeta_score(y_test, y_pred, beta=0.5)

# ✅ Cross-Entropy Loss (also called Log Loss)
logloss = log_loss(y_test, y_pred)

# Display results
print(f"F2 Score (Recall-weighted):     {f2:.3f}")
print(f"F0.5 Score (Precision-weighted): {f05:.3f}")
# print(f"Cross-Entropy Loss (Log Loss):   {logloss:.3f}")

# ROC Curve and AUC
y_pred_prob = model.predict_proba(X_test)[:, 1]
roc_auc = roc_auc_score(y_test, y_pred_prob)
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
# print(roc_auc)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.show()


























































































