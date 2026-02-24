from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import json
from src.gemstones.pipeline.predict_pipeline import CustomData, PredictPipeline
import pandas as pd
from typing import Annotated, Literal, Optional

MODEL_VERSION = '0.0.1'

app = FastAPI()

class Data(BaseModel):
    carat: Annotated[float, Field(..., description = 'carat of your stone')]
    cut: Annotated[str, Field(..., description = 'cut quality of the stone')]
    color: Annotated[str, Field(..., description = 'color of the stone')]
    clarity: Annotated[str, Field(..., description = 'clarity of the stone')]
    depth: Annotated[float, Field(..., description = 'depth of the stone')]
    table: Annotated[float, Field(..., description = 'table of the stone')]
    x: Annotated[str, Field(..., description = 'dimensions of stone X')]
    y: Annotated[str, Field(..., description = 'dimensions of stone Y')]
    z: Annotated[str, Field(..., description = 'dimensions of stone Z')]

@app.get('/')
def home():
    return {'message': 'homepage'}

@app.get('/health')
def health():
    return {
        'status': 'OK',
        'MODEL_VERSION': MODEL_VERSION
    }

@app.post('/predict')
def predict(input: Data):
    data = CustomData(
        carat = input.carat,
        cut = input.cut,
        color = input.color,
        clarity = input.clarity,
        table = input.table,
        x = input.x,
        y = input.y,
        z = input.z
    )

    df = data.get_dataframe()
    pred = PredictPipeline()
    prediction = pred.predict(df)

    return JSONResponse(status_code = 200, content = {'predicted': prediction.tolist()})