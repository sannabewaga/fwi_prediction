from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler

# Initialize Flask app
application = Flask(__name__)
app = application

# Load trained Ridge model and StandardScaler
ridge = pickle.load(open('models/ridge.pkl', 'rb'))
scaler = pickle.load(open('models/scaler.pkl', 'rb'))

# Home route
@app.route('/')
def index():
    return render_template('index.html')

# Predict route
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            # Extract input values from the form
            temperature = float(request.form.get('Temperature'))
            rh = float(request.form.get('RH'))
            ws = float(request.form.get('Ws'))
            rain = float(request.form.get('Rain'))
            ffmc = float(request.form.get('FFMC'))
            dmc = float(request.form.get('DMC'))
            dc = float(request.form.get('DC'))
            isi = float(request.form.get('ISI'))
            bui = float(request.form.get('BUI'))

            # Convert to NumPy array and reshape for the model
            input_data = np.array([[temperature, rh, ws, rain, ffmc, dmc, dc, isi, bui]])

            # Scale the input data
            scaled_data = scaler.transform(input_data)

            # Predict using the trained Ridge model
            prediction = ridge.predict(scaled_data)

            # Pass the prediction result to the template
            return render_template('home.html', result=prediction[0])

        except Exception as e:
            return jsonify({"error": str(e)})

    # If GET request, just show the form
    return render_template('home.html', result="")

# Run the Flask app
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
