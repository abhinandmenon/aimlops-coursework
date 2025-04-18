import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, ValidationError

from customer_churn_model.config.core import config
from customer_churn_model.processing.data_manager import pre_pipeline_preparation


def validate_inputs(*, input_df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[dict]]:
    """Check model inputs for unprocessable values."""

    pre_processed = pre_pipeline_preparation(data_frame=input_df)
    validated_data = pre_processed[config.model_config_.features].copy()
    errors = None

    try:
        # replace numpy nans so that pydantic can validate
        MultipleDataInputs(
            inputs=validated_data.replace({np.nan: None}).to_dict(orient="records")
        )
    except ValidationError as error:
        errors = error.json()

    return validated_data, errors


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
