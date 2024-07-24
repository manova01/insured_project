



# Insurance Model Training and Deployment Pipeline

## Project Overview

This project encompasses a complete machine learning workflow for an insurance dataset. The workflow includes data loading, cleaning, preprocessing, model training, deployment, and monitoring. The project leverages several tools and technologies to ensure best practices are followed, including MLflow for experiment tracking, Prefect for workflow orchestration, Flask and Docker for deployment, and Evidently for monitoring.

## Technologies Used

- **Cloud**: Can be deployed on AWS, GCP, Azure, or any cloud platform.
- **Experiment Tracking**: MLflow
- **Workflow Orchestration**: Prefect
- **Monitoring**: Evidently
- **CI/CD**: GitHub Actions
- **Infrastructure as Code (IaC)**: Terraform (optional for cloud resource provisioning)

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/manova01/insurance_pro.git
cd insurance_pro
```

### 2. Create and Activate a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set Up MLflow Tracking

Ensure that MLflow is installed and running:

```bash
pip install mlflow
mlflow ui
```

### 5. Configure Prefect

Install Prefect:

```bash
pip install prefect
```

Start the Prefect server:

```bash
prefect server start
```

Configure Prefect to communicate with the server:

```bash
prefect config set PREFECT_API_URL=http://127.0.0.1:4200/api
```

### 6. Create Prefect Flow

The Prefect flow script `model_training_flow.py` is already included in the repository. It orchestrates the entire model training pipeline.

### 7. Flask Application for Model Deployment

The Flask app provides a REST API for model predictions.

Create a `Dockerfile` to containerize the Flask application:

```dockerfile
FROM python:3.8-slim

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY . .

CMD ["python", "app.py"]
```

Build and run the Docker container:

```bash
docker build -t insurance-model .
docker run -p 5000:5000 insurance-model
```

### 8. Model Monitoring with Evidently

Create a monitoring script `monitor.py` to track model performance:

```python
from evidently.dashboard import Dashboard
from evidently.dashboard.tabs import DataDriftTab, RegressionPerformanceTab
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split

# Load data
url = 'https://raw.githubusercontent.com/manova01/insurance_pro/main/insurance%20data.csv'
df = pd.read_csv(url)

# Data preprocessing
df.columns = df.columns.str.lower().str.replace(' ', '_')
df = df.dropna()

# Split the data
df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=42)

# Load model
model = joblib.load('model.pkl')

# Make predictions
X_test = df_test.drop(columns='charges')
y_test = df_test['charges']
y_pred = model.predict(X_test)

# Create the dashboard
dashboard = Dashboard(tabs=[DataDriftTab(), RegressionPerformanceTab()])
dashboard.calculate(df_train, df_test)

# Save the dashboard
dashboard.save("evidently_dashboard.html")
```

### 9. CI/CD with GitHub Actions

Set up a GitHub Actions workflow to automate testing and deployment. Create a `.github/workflows/main.yml` file:

```yaml
name: CI/CD Pipeline

on:
  push:
    branches:
      - main

jobs:
  build:
    runs-on: ubuntu-latest

    steps:
    - name: Checkout code
      uses: actions/checkout@v2

    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.8

    - name: Install dependencies
      run: |
        python -m venv venv
        source venv/bin/activate
        pip install -r requirements.txt

    - name: Run tests
      run: |
        source venv/bin/activate
        pytest

    - name: Build and push Docker image
      uses: docker/build-push-action@v2
      with:
        context: .
        push: true
        tags: username/insurance-model:latest
```

Replace `username/insurance-model` with your Docker Hub repository name.

## Usage

### Model Training Pipeline

1. Start the Prefect server.
2. Register the Prefect flow:

    ```bash
    python model_training_flow.py
    ```

3. Run the flow:

    ```bash
    prefect deployment run model_training_pipeline
    ```

### Model Deployment

1. Build and run the Docker container for the Flask app:

    ```bash
    docker build -t insurance-model .
    docker run -p 5000:5000 insurance-model
    ```

2. Make a prediction by sending a POST request to `http://localhost:5000/predict` with a JSON payload.

### Monitoring

Run the monitoring script to generate a performance dashboard:

```bash
python monitor.py
```

### CI/CD

Push changes to the `main` branch to trigger the GitHub Actions workflow.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request.

## License

This project is licensed under the MIT License.

---

By following this README, you should be able to set up and run the entire machine learning workflow for the insurance dataset, ensuring best practices and leveraging modern tools and technologies.
