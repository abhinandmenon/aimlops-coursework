from typing import Any, List, Optional
from datetime import date

from pydantic import BaseModel


class PredictionResults(BaseModel):
    errors: Optional[Any]
    version: str
    predictions: Optional[float]


class DataInputSchema(BaseModel):
    dteday: Optional[date]
    season: Optional[str]
    hr: Optional[str]
    holiday: Optional[str]
    weekday: Optional[str]
    workingday: Optional[str]
    weathersit: Optional[str]
    temp: Optional[float]
    atemp: Optional[float]
    hum: Optional[float]
    windspeed: Optional[float]


class MultipleDataInputs(BaseModel):
    inputs: List[DataInputSchema]
