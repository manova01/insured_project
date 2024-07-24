import pytest
import pandas as pd
from sklearn.linear_model import Ridge

def test_model_training():
    # Sample test for training
    data = {'age': [25], 'bmi': [30.0], 'children': [1], 'claim_amount': [10000],
            'past_consultations': [1], 'num_of_steps': [1000000], 'hospital_expenditure': [500000],
            'number_of_past_hospitalizations': [1], 'anual_salary': [50000], 'region_southeast': [1]}
    df = pd.DataFrame(data)
    X = df
    y = [1500]  # Sample target value

    model = Ridge()
    model.fit(X, y)
    
    assert model.coef_ is not None
