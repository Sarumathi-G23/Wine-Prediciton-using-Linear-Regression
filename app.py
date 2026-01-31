from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
model = pickle.load(open("model.pkl", "rb"))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get values from form
        t1 = float(request.form['t1'])
        t2 = float(request.form['t2'])
        t3 = float(request.form['t3'])

        # Convert to numpy array
        features = np.array([[t1, t2, t3]])

        # Prediction
        prediction = model.predict(features)[0]

        return render_template(
            'result.html',
            prediction=f"The predicted wine quality is: {prediction}"
        )

    except Exception as e:
        return f"Error: {e}"

if __name__ == "__main__":
    app.run(debug=True)
