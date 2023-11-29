from flask import Flask, request, render_template, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load your trained model
model = joblib.load('random_forest_model.joblib')  # Update with your model path


@app.route('/', methods=['GET'])
def home():
    return render_template('form.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Collect data from form
            age = request.form.get('age')
            hypertension = request.form.get('hypertension')
            heart_disease = request.form.get('heart_disease')
            # Add other feature data collection here

            # Prepare the feature array for prediction
            feature_array = np.array([[age, hypertension, heart_disease]])  # Update with all features

            # Make prediction
            prediction = model.predict(feature_array)
            return jsonify({'prediction': str(prediction[0])})

        except Exception as e:
            return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
