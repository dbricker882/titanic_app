import json
import requests

data = {"features": [5,3,0,0,8.05,35.0,True,False,False,True]}

url = "http://127.0.0.1:8000/predict/"

# data = json.dumps(data)
response = requests.post(url, json = data)
# print(response.json())

print(f"Status Code: {response.status_code}")
print(f"Response Text: '{response.text}'")