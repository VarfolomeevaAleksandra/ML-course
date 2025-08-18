import joblib
import os
import json

import pandas as pd
from pydantic import BaseModel
import glob

path = os.environ.get('PROJECT_PATH', '.')

models_dir = os.path.join(path, "data", "models")
model_files = glob.glob(os.path.join(models_dir, "cars_pipe_*.pkl"))
model_path = max(model_files, key=os.path.getctime)  # берём самый новый
model = joblib.load(model_path)

#model_path = os.path.join(path, "data", "models", "cars_pipe_202508182007.pkl")


class Form(BaseModel):
    description: str
    fuel: str
    id: int
    image_url: str
    lat: float
    long: float
    manufacturer: str
    model: str
    odometer: float
    posting_date: str
    price: int
    region: str
    region_url: str
    state: str
    title_status: str
    transmission: str
    url: str
    year: float


class Prediction(BaseModel):
    id: int
    pred: str
    price: int


test_folder_path = os.path.join(path, "data", "test")


def read_file(filename):
    if filename.endswith('.json'):
        file_path = os.path.join(test_folder_path, filename)

        # Загружаем JSON
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            form_obj = Form(**data)
    return form_obj


results = []


def predict():
    import modules.pipeline
    with open(model_path, 'rb') as file:
        model = joblib.load(file)
    for filename in os.listdir(test_folder_path):
        data = read_file(filename)
        df = pd.DataFrame.from_dict([data.model_dump()])
        y = model.predict(df)
        results.append({
            "id": data.id,
            "pred": y[0],
            "price": data.price
        })
    predict_path = os.path.join(path, "data", "predictions", "predictions.csv")
    prediction = pd.DataFrame(results)
    prediction.to_csv(predict_path, index=False, encoding="utf-8")

if __name__ == '__main__':
    predict()
