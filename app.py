from flask import Flask, request
from joblib import load

app = Flask(__name__)
model = load('SeattleModel.pkl')

# GET REQUEST
@app.route('/')
def welcome():
    return "Welcome All"


# GET REQUEST
@app.route('/predict')
def predict_survival():

    precipitation = request.args.get("precipitation")
    tempMax = request.args.get("tempMax")
    tempMin = request.args.get("tempMin")
    wind = request.args.get("wind")

    arrayTest = [precipitation, tempMax, tempMin, wind ]
    prediction = model.predict([arrayTest])
    return "the predicted value is" + str(prediction)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
