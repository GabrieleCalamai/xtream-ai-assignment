import requests
import pandas as pd


# Data for prediction
data = pd.read_csv('X_test_2.csv')
print(data.shape)

input_data_array = data.values

# Send POST request to the API
response = requests.post('http://localhost:5000/predict', json = {'data': input_data_array.tolist()})

if response.status_code == 200:
    try:
        predictions = response.json()['predictions']
        print('Predictions:', predictions)
    except Exception as e:
        print('Error:', response.text)
else:
    print('Error:', response.text)