def test_predict(client):
    test_input = {
        "age": 45.0,
        "sex": "male",
        "bmi": 28.700,
        "children": 2.0,
        "smoker": "no",
        "claim_amount": 32993.77432,
        "past_consultations": 16.0,
        "num_of_steps": 902022.0,
        "hospital_expenditure": 8.640895e+06,
        "number_of_past_hospitalizations": 1.0,
        "anual_salary": 9.436591e+07,
        "region": "southwest"
    }
    response = client.post('/predict', json=test_input)
    assert response.status_code == 200
    assert 'prediction' in response.json
