#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
import joblib
import pickle

# Load the data
url = 'https://raw.githubusercontent.com/manova01/insured_project/main/insurance%20data.csv'
df = pd.read_csv(url)
df.columns = df.columns.str.lower().str.replace(' ', '_')

# Display data types and unique values
for col in df.columns:
    print(col)
    print(df[col].unique()[:5])
    print(df[col].nunique())
    print()

# Identify object type columns
strings = list(df.dtypes[df.dtypes=='object'].index)

# EDA
print(df.info())
print(df.describe())

# Histograms
df.hist(bins=50, figsize=(20, 15))
plt.show()

# Pairplot
sns.pairplot(df)
plt.show()

# Plot charges
sns.histplot(df.charges)
plt.show()

# Check for null values
print(df.isnull().sum())

# Drop rows with null values
df = df.dropna()

# Check for null values again
print(df.isnull().sum())

# Split the data into train, validation, and test sets
df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=42)

# Reset index
df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

# Separate target variable
y_train = df_train.charges.values
y_val = df_val.charges.values
y_test = df_test.charges.values

# Remove target variable from feature set
del df_train['charges']
del df_val['charges']
del df_test['charges']

# One Hot Encoding
train_dicts = df_train.to_dict(orient='records')
dv = DictVectorizer(sparse=False)
X_train = dv.fit_transform(train_dicts)

val_dicts = df_val.to_dict(orient='records')
X_val = dv.transform(val_dicts)

# Hyperparameter tuning with Ridge regression
param_grid = {
    'alpha': [0.1, 1.0, 10.0, 100.0],
    'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']
}
model = Ridge()
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

# Best model from GridSearchCV
best_model = grid_search.best_estimator_

# Evaluate the model
y_pred = best_model.predict(X_val)
mse = mean_squared_error(y_val, y_pred)
rmse = np.sqrt(mse)
print(f'Mean Squared Error: {mse}')
print(f'Root Mean Squared Error: {rmse}')

# Residual plot
residuals = y_val - y_pred
sns.histplot(residuals, kde=True)
plt.xlabel('Residuals')
plt.title('Residuals Distribution')
plt.show()

# Scatter plot of actual vs predicted values
plt.scatter(y_val, y_pred)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted Values')
plt.show()

# Save the model using joblib
joblib.dump(best_model, "ridge_model.pkl")

# Save the model using pickle
with open('ridge_model.pkl', 'wb') as file:
    pickle.dump(best_model, file)
