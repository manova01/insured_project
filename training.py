import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import joblib
import pickle
import mlflow
import mlflow.sklearn
from prefect import task, flow
from prefect.deployments import Deployment
from prefect.filesystems import GitHub
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from evidently.test_suite import TestSuite
from evidently.test_preset import DataDriftTestPreset

# Load the data
@task
def load_data(url: str):
    df = pd.read_csv(url)
    df.columns = df.columns.str.lower().str.replace(' ', '_')
    return df

# Preprocess the data
@task
def preprocess_data(df: pd.DataFrame):
    df = df.dropna()
    df = df.reset_index(drop=True)
    return df

# Split the data
@task
def split_data(df: pd.DataFrame):
    df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
    df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=42)
    y_train = df_train.charges.values
    y_val = df_val.charges.values
    y_test = df_test.charges.values
    df_train = df_train.drop(columns=['charges'])
    df_val = df_val.drop(columns=['charges'])
    df_test = df_test.drop(columns=['charges'])
    return df_train, df_val, df_test, y_train, y_val, y_test

# Vectorize the data
@task
def vectorize_data(df_train, df_val):
    dv = DictVectorizer(sparse=False)
    train_dicts = df_train.to_dict(orient='records')
    val_dicts = df_val.to_dict(orient='records')
    X_train = dv.fit_transform(train_dicts)
    X_val = dv.transform(val_dicts)
    return X_train, X_val, dv

# Scale the data
@task
def scale_data(X_train, X_val):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    return X_train_scaled, X_val_scaled, scaler

# Apply PCA
@task
def apply_pca(X_train_scaled, X_val_scaled):
    pca = PCA(n_components=0.95)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_val_pca = pca.transform(X_val_scaled)
    return X_train_pca, X_val_pca, pca

# Train the model
@task
def train_model(X_train_pca, y_train):
    param_grid = {
        'alpha': [0.1, 1.0, 10.0, 100.0],
        'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']
    }
    model = Ridge()
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X_train_pca, y_train)
    best_model = grid_search.best_estimator_
    return best_model, grid_search.best_params_

# Evaluate the model
@task
def evaluate_model(model, X_val_pca, y_val):
    y_pred = model.predict(X_val_pca)
    mse = mean_squared_error(y_val, y_pred)
    rmse = np.sqrt(mse)
    return mse, rmse

# Log model and metrics to MLflow
@task
def log_to_mlflow(model, mse, rmse, params):
    mlflow.log_params(params)
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("rmse", rmse)
    mlflow.sklearn.log_model(model, "ridge_model")

# Save the model and vectorizer
@task
def save_model_and_vectorizer(model, dv):
    joblib.dump(model, "ridge_model.pkl")
    joblib.dump(dv, "vectorizer.pkl")

# Generate and log Evidently reports
@task
def generate_evidently_report(df_train, df_val, y_train, y_val, X_val_pca, model):
    # Column mapping
    column_mapping = ColumnMapping(
        prediction='charges',
        target='charges',
        numerical_features=['charges']
    )

    # Data drift report
    data_drift_report = Report(metrics=[DataDriftPreset()])
    data_drift_report.run(reference_data=df_train, current_data=df_val, column_mapping=column_mapping)
    data_drift_report.save_html('data_drift_report.html')

    # Model performance report
    test_suite = TestSuite(tests=[DataDriftTestPreset()])
    test_suite.run(reference_data=df_train, current_data=df_val, column_mapping=column_mapping)
    test_suite.save_html('model_performance_report.html')

# Define the Prefect flow
@flow(name="insurance_model_training")
def insurance_model_training():
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
    generate_evidently_report(df_train, df_val, y_train, y_val, X_val_pca, model)

# Register and run the flow
if __name__ == "__main__":
    insurance_model_training()

    # Deployment to GitHub (optional, if needed for version control and remote execution)
    github_block = GitHub.load("insured-project-github-repo")
    deployment = Deployment.build_from_flow(
        flow=insurance_model_training,
        name="insurance_model_training_deployment",
        storage=github_block,
    )
    deployment.apply()
