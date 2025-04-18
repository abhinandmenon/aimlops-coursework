from typing import Any, List, Optional
from datetime import date

from pydantic import BaseModel, Field


class PredictionResults(BaseModel):
    errors: Optional[Any]
    version: str
    predictions: Optional[float]


class DataInputSchema(BaseModel):
    CustomerID: Optional[int]
    Age: Optional[int]
    Gender: Optional[str]
    Tenure: Optional[int]
    Usage_Frequency: Optional[int] = Field(alias='Usage Frequency')
    Support_Calls: Optional[int] = Field(alias='Support Calls')
    Payment_Delay: Optional[int] = Field(alias='Payment Delay')
    Subscription_Type: Optional[str] = Field(alias='Subscription Type')
    Contract_Length: Optional[str] = Field(alias='Contract Length')
    Total_Spend: Optional[float] = Field(alias='Total Spend')
    Last_Interaction: Optional[int] = Field(alias='Last Interaction')


class MultipleDataInputs(BaseModel):
    inputs: List[DataInputSchema]
