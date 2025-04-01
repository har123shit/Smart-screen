# " IOT based smart screen control using handgesture"
# Flask app
from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

# Load data and pre-trained model
data = pd.read_csv('CleanedHouseData.csv')
pipe = pickle.load(open("house_price_prediction_model.pkl", "rb"))


@app.route('/')
def home():
    # Extract unique locations
    locations = data['location'].unique()
    locations_list = locations.tolist()  # Convert ndarray to list
    return render_template('index.html', locations=locations_list)


@app.route('/predict', methods=['POST'])
def predict():
    # Parse JSON data from the request body
    data = request.json
    location = data.get('location')
    bhk = float(data.get('bhk'))
    bath = float(data.get('bath'))
    total_sqft = float(data.get('total_sqft'))

    # Check if any of the input values are missing or invalid
    if not all([location, bhk, bath, total_sqft]):
        return "Error: Please provide all required input values"

    try:
        # Predict the price using the pipeline
        prediction = pipe.predict(pd.DataFrame({'location': [location], 'total_sqft': [total_sqft], 'bath': [bath], 'bhk': [bhk]}))[0]
        return str(prediction*100000)
    except Exception as e:
        return f"Error: Failed to make prediction. {e}"


if __name__ == '__main__':
    app.run(debug=True)
