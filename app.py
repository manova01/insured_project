from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load the model and vectorizer
model = joblib.load('linear_regression_model.pkl')
dv = joblib.load('vectorizer.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    df = pd.DataFrame([data])
    
    # Ensure that data keys match the model's feature names
    X = dv.transform(df.to_dict(orient='records'))
    prediction = model.predict(X)
    
    return jsonify({'prediction': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)
