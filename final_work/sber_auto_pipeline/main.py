import traceback
import dill

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional



app = FastAPI()
with open('model/sberauto_pipe.pkl', 'rb') as file:
    model = dill.load(file)


class Form(BaseModel):
    utm_source: Optional[str] = None
    utm_medium: Optional[str] = None
    utm_campaign: Optional[str] = None
    utm_adcontent: Optional[str] = None
    utm_keyword: Optional[str] = None
    device_category: Optional[str] = None
    device_os: Optional[str] = None
    device_brand: Optional[str] = None
    device_model: Optional[str] = None
    device_screen_resolution: Optional[str] = None
    device_browser: Optional[str] = None
    geo_country: Optional[str] = None
    geo_city: Optional[str] = None


class Prediction(BaseModel):
    pred: int


@app.get('/status')
def status():
    return "I'm OK"


@app.get('/version')
def version():
    return model['metadata']


@app.post('/predict', response_model=Prediction)
def predict(form: Form):
    df = pd.DataFrame.from_dict([form.model_dump()])
    try:
        y = model['model'].predict_proba(df)
        result = (y[:,1] >= 0.0344).astype(int)
        return {
            'pred': result[0]
        }
    except Exception as e:
        return {
            'pred': f"{type(e).__name__}: {e}\n{traceback.format_exc()}",
        }