import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import joblib
import mlflow
import mlflow.sklearn

# Define functions as you have in your code

def train_and_log_model():
    url = 'https://raw.githubusercontent.com/manova01/insured_project/main/insurance%20data.csv'
    df = load_data(url)
    df_clean = preprocess_data(df)
    df_train, df_val, df_test, y_train, y_val, y_test = split_data(df_clean)
    X_train, X_val, dv = vectorize_data(df_train, df_val)
    X_train_scaled, X_val_scaled, scaler = scale_data(X_train, X_val)
    X_train_pca, X_val_pca, pca = apply_pca(X_train_scaled, X_val_scaled)
    model, params = train_model(X_train_pca, y_train)
    mse, rmse = evaluate_model(model, X_val_pca, y_val)
    log_to_mlflow(model, mse, rmse, params)
    save_model_and_vectorizer(model, dv)

if __name__ == "__main__":
    train_and_log_model()
