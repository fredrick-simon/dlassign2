from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np

app = Flask(__name__)

# Load the saved ANN model
model = tf.keras.models.load_model("1.h5")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    country = int(request.form['country'])
    store = int(request.form['store'])
    product = int(request.form['product'])

    # Define a function to preprocess the input values and make a prediction
    def predict_sales(country, store, product):
        # Preprocess the input values
        inputs = np.array([country, store, product]).reshape(1, -1)

        # Make a prediction using the loaded model
        prediction = model.predict(inputs)

        return prediction[0][0]

    prediction = predict_sales(country, store, product)
    return f"The predicted sales for this product is {prediction:.2f}."

if __name__ == '__main__':
    app.run(debug=True)
