import json

import dill

import pandas as pd

with open('model/car_pipe.pkl','rb') as file:
    model = dill.load(file)

with open('model/data/7316509996.json', 'r') as f:
    data = json.load(f)

print(data)

form = pd.DataFrame([data]).drop(labels='id', axis=1)
y = model['model'].predict(form)

print(y)