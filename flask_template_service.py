from flask import Flask, request
import pandas as pd
import argparse
import pickle
import numpy as np
from sklearn.linear_model import LinearRegression

app = Flask(__name__)


def read_arguments():
    """
    [summary]
    Reads the console arguments
    Returns:
        [string]: [The path of the pickle model file]
    """
    # Creating a parser for the arguments
    parser = argparse.ArgumentParser(
        description='Get the pickle model file path')
    # Adding the file path argument
    parser.add_argument('-path', help='pickle model file path', required=True)
    args = parser.parse_args()
    # returning the file path
    return args.path


@app.route('/')
def hello():
    """
    Returns a message to the users telling them how to user the requests
    """
    return "Welcome, use this query to predict the price : /predict-price?transaction_date=2012.917&house_age=19.5&nearest_distance=306.5947&num_convenience_stores=9&latitude=24.98034&longitude=121.53951"


@app.route('/predict-price', methods=["GET"])
def predict():
    """
    [summary]
    The functions uses the model from the pickle file to predict the price of the houses
    Returns:
        [string]: [The predicted price of the house from the user request]
    """
    # This will tell the user if you didn't send all the features to the service
    if len(request.args) < 6:
        return "Please make sure you added all the house specification which are: transaction_date, house_age, nearest_distance, num_convenience_stores, latitude, longitude"
    # Storing the arguments values in variables
    transaction_date = float(request.args.get('transaction_date'))
    house_age = float(request.args.get('house_age'))
    nearest_distance = float(request.args.get('nearest_distance'))
    num_convenience_stores = float(request.args.get('num_convenience_stores'))
    latitude = float(request.args.get('latitude'))
    longitude = float(request.args.get('longitude'))
    # Adding the features to a 2D matrix to work with the pickle model
    house_features = np.array([transaction_date, house_age, nearest_distance,
                               num_convenience_stores, latitude, longitude]).reshape(1, 6)
    # Using the predict model of the pickle to predict the price
    predicted_price = loaded_model.predict(house_features)
    # returning the predicted price to the user
    return "The predicted price is: "+str(predicted_price)


if __name__ == '__main__':
    # The path of the pickle file which will be entered from the console
    file_path = read_arguments()
    # Loading the model using Scikit-learn library
    loaded_model = pickle.load(open(file_path, 'rb'))
    app.run(debug=True, use_reloader=False)
