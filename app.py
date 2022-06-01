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
    wind = request.args.get("wind")
    tempMax = request.args.get("tempMax")

    arrayTest = [-0.45068983, 0.49293002, 0.80153523, wind]
    #prediction = model.predict([[int(experience)]])
    prediction = model.predict([arrayTest])
    return "the predicted value is" + str(prediction)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
