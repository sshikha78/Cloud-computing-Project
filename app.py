from flask import Flask, request, render_template
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model
model = joblib.load('random_forest_model.joblib')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract input data from the form
    age = request.form['age']
    hypertension = request.form['hypertension']
    heart_disease = request.form['heart_disease']

    # Create a DataFrame from the input data
    input_data = pd.DataFrame({
        'age': [age],
        'hypertension_0': [1 if hypertension == 'False' else 0],
        'hypertension_1': [1 if hypertension == 'True' else 0],
        'heart_disease_0': [1 if heart_disease == 'False' else 0],
        'heart_disease_1': [1 if heart_disease == 'True' else 0]
    })

    # Make a prediction
    prediction = model.predict(input_data)

    # Convert prediction to a readable format
    prediction_text = 'High Risk of Stroke' if prediction[0] == 1 else 'Low Risk of Stroke'

    return render_template('result.html', prediction=prediction_text)

if __name__ == '__main__':
    app.run(debug=True)
