from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load models
with open('landslide_models.pkl', 'rb') as f:
    models = pickle.load(f)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input values from form
        rainfall = float(request.form['rainfall'])
        slope = float(request.form['slope'])
        moisture = float(request.form['moisture'])
        elevation = float(request.form['elevation'])
        veg_cover = float(request.form['veg_cover'])
        model_name = request.form['model_name']

        # Create input array
        input_data = np.array([[rainfall, slope, moisture, elevation, veg_cover]])
        
        # Select and use model
        model = models.get(model_name)
        if model_name == 'linear_regression':
            prediction = model.predict(input_data)[0]
            # Convert continuous to 0 or 1 for unified result
            result = 1 if prediction > 0.5 else 0
            score = round(float(prediction) * 100, 2)
        else:
            prediction = model.predict(input_data)[0]
            result = int(prediction)
            score = 100 if result == 1 else 0

        risk_label = "High Risk" if result == 1 else "Low Risk"
        risk_class = "high-risk" if result == 1 else "low-risk"

        return render_template('index.html', 
                               prediction_text=f'Predicted {risk_label} ({score}%)', 
                               risk_class=risk_class,
                               rainfall=rainfall,
                               slope=slope,
                               moisture=moisture,
                               elevation=elevation,
                               veg_cover=veg_cover)

    except Exception as e:
        return f"Error: {e}"

if __name__ == "__main__":
    app.run(debug=True)
