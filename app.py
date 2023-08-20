from flask import Flask, request, jsonify
import sklearn
import joblib
app = Flask(__name__)

# loading model
model = joblib.load('trained_model.pkl')

@app.route('/')
def hello_world():  # put application's code here
    return 'Hello World!'

@app.route('/predict', methods = ['POST'])
def predict():
    try:
        data = request.json['data']
        predictions = model.predict(data)
        return jsonify({'predictions': predictions.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run(host = '0.0.0.0', port = 5000)
