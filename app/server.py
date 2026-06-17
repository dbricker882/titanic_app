from fastapi import FastAPI
import joblib
import numpy as np

model = joblib.load('app/titanic_rforest.joblib')

class_names = np.array(model.classes_)

app = FastAPI()

@app.get('/')
def reed_root():
    return {'message': 'Titanic model API'}

@app.post('/predict')
def predict(data: dict):
    features = np.array(data['features']).reshape(1, -1)
    prediction = model.predict([features])
    class_name = class_names[prediction][0]
    return {'prediction': class_name}

