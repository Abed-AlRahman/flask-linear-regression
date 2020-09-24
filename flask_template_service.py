from flask import Flask, request
import pandas as pd

app = Flask(__name__)

@app.route('/')
def hello():
    return "Welcome, use this query to predict the price : /predict-price?transaction_date=2012.917&house_age=19.5&nearest_distance=306.5947&num_convenience_stores=9&latitude=24.98034&longitude=121.53951"

@app.route('/predict-price', methods=["GET"])
def predict():

    transaction_date = float(request.args.get('transaction_date'))
    house_age = float(request.args.get('house_age'))
    nearest_distance = float(request.args.get('nearest_distance'))
    num_convenience_stores = float(request.args.get('num_convenience_stores'))
    latitude = float(request.args.get('latitude'))
    longitude = float(request.args.get('longitude'))

    predicted_price = thetazero + thetas[0]*transaction_date + thetas[1]*house_age + thetas[2]*nearest_distance + thetas[3]*num_convenience_stores + thetas[4]*latitude + thetas[5]*longitude
    
    return "The predicted price is: "+str(predicted_price)

if __name__ == '__main__':

    data = pd.read_csv("lm_parameters.csv") 

    thetazero = float(data.iloc[0][0])

    thetas = data.columns

    thetas = [float(theta) for theta in thetas]
    app.run(debug=True, use_reloader=False)