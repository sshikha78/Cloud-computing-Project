#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from imblearn.over_sampling import SMOTE
import imblearn
from sklearn.neighbors import KNeighborsClassifier
# from xgboost import XGBClassifier
from collections import Counter
from imblearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder

#%%
df = pd.read_csv("healthcare-dataset-stroke-data.csv")

#%%
# Print the first 5 rows of the DataFrame
print(df.head().to_string())
# Print the shape of the DataFrame (number of rows and columns)
print(df.shape)
# Check for missing values in the DataFrame
print(df.isna().sum())
# Found 201 NULL values in bmi column
# Get summary statistics for the numerical columns in the DataFrame
print(df.describe())
# Get information about the DataFrame, including column data types and number of non-null values
print(df.info())
# Print the data types of each column in the DataFrame
print(df.dtypes)
# We must check that there are no unexpected unique values in each column
for col in df.columns:
  if df[col].dtype != 'float64':
    print(f"{col} has unique values:{df[col].unique()}")
# Filling Null Data
# We will fill the null values of bmi column with the mean of this column
df['bmi'].fillna(value=df['bmi'].mean(),inplace=True)
print("after filling null values:",df.isna().sum())
# Drop the ID column as it is not relevant for modeling
df.drop('id', axis=1, inplace=True)
# Convert the stroke column to integers (0 for no stroke, 1 for stroke)
df['stroke'] = df['stroke'].astype(int)

#%%
# Target feature - Stroke
print("Value count in the stroke : \n",df['stroke'].value_counts())

# Encode categorical variables
df = pd.get_dummies(df, columns=['gender','hypertension', 'heart_disease', 'ever_married', 'work_type', 'Residence_type','smoking_status'])

#%%
# Feature Engineering:
df = df[['age', 'hypertension_0', 'hypertension_1', 'heart_disease_0', 'heart_disease_1', 'stroke']]
# df = df[['age', 'hypertension', 'heart_disease', 'stroke']]
X = df.drop(['stroke'], axis=1)
y = df['stroke']
# # SPLIT TEST AND TRAIN PART
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
plt.figure(figsize=(10, 5))
plt.title("Class Distribution before SMOTE")
plt.hist(y, bins=2, rwidth=0.8)
plt.xticks([0, 1])
plt.xlabel("Stroke(0=No,1=Yes)")
plt.ylabel("Count")
plt.show()
unique, counts = np.unique(y_train, return_counts=True)
#%%
# Print the count of instances in each class before oversampling
print("Class counts before SMOTE oversampling:")
for i in range(len(unique)):
    print("Class", unique[i], ":", counts[i])
#%%
# Apply SMOTE oversampling
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)


# Count the number of instances in each class after oversampling
unique, counts = np.unique(y_train, return_counts=True)
#%%
# Print the count of instances in each class after oversampling
print("Class counts after SMOTE oversampling:")
for i in range(len(unique)):
    print("Class", unique[i], ":", counts[i])
plt.figure(figsize=(10, 5))
plt.title("Class Distribution after SMOTE")
plt.hist(y_train, bins=2, rwidth=0.8)
plt.xticks([0, 1])
plt.xlabel("Stroke(0=No,1=Yes)")
plt.ylabel("Count")
plt.legend()
plt.show()

# Perform feature scaling 
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
#%%
# MACHINE LEARNING ALGORITHMS
# RANDOM FOREST
rfc = RandomForestClassifier(random_state=42)
rfc.fit(X_train_scaled, y_train)
y_pred = rfc.predict(X_test_scaled)
print("random-forest Classification report \n",classification_report(y_test, y_pred))
y_pred_proba = rfc.predict_proba(X_test_scaled)[:, 1]


# %%

from sklearn.ensemble import RandomForestClassifier
import joblib

# Train your Random Forest model
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Save the model to a file
joblib.dump(model, 'random_forest_model.joblib')

#%%
# Load the model
# loaded_model = joblib.load('random_forest_model.joblib')


# %%
