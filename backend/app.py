from fastapi import FastAPI
from datetime import datetime, timedelta
from typing import List
from pydantic import BaseModel


from fastapi.encoders import jsonable_encoder

from fastapi.middleware.cors import CORSMiddleware

from utility.utilities import clean_data, get_features, feature_selection, preprocess_data ,get_label, get_model_by_label

import json
import pandas as pd

app = FastAPI()

origins = [
    "http://localhost:3000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#Pydantic
class DataItem(BaseModel):
    point_timestamp: str
    point_value: float

class DataInput(BaseModel):
    data: List[DataItem]

@app.get("/")
def root ():
  return {"message": "Hello World!"}


@app.post("/predict/")
async def get_data(date_from: str, date_to: str, period: int, data_input: DataInput):
    print(data_input)
    output = {}

    # Convert data_input to DataFrame
    data_input = json.dumps(data_input.dict())
    data_list = json.loads(data_input)
    df = pd.DataFrame(data_list['data'])

    #Clean the data and extract features
    df = clean_data(df)

    #Extract Time Series features
    features_df = get_features(df.copy())
    
    #Feature Extraction
    features_df = feature_selection(features_df)
    features_df = preprocess_data(features_df)

    #Fit the TSA Model and collect required data
    label = get_label(features_df)
    name, tsa_model = get_model_by_label(df, date_from, date_to, label)
    
    tsa_model.fit_model()
    predictions = tsa_model.make_predictions()
    mape = tsa_model.get_mape()
    forecast = tsa_model.get_forecast(period)
    
    #Format the output
    output["model"] = name
    output["mape"] = mape

    output["result"] = []

    mask = df['ds'].isin(predictions['ds'])
    matching_y = df.loc[mask, 'y']

    for row, y in zip(predictions.iterrows(), matching_y):
        t = {}

        t["point_timestamp"] = row[1]["ds"]
        t["yhat"] = row[1]["yhat"]
        t["y"] = y

        output["result"].append(t)
    
    output["forecast"] = []

    for _, row in forecast.iterrows():
        t = {}

        t["point_timestamp"] = row["ds"]
        t["yhat"] = row["yhat"]

        output["forecast"].append(t)

    print(output)

    return output