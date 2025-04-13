import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from pydantic import BaseModel, ValidationError

from bikeshare_model.config.core import config
from bikeshare_model.processing.data_manager import pre_pipeline_preparation


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
    Date: Optional[str]
    Season: Optional[str]
    Hour: Optional[str]
    Holiday: Optional[str]
    Weekday: Optional[str]
    Workingday: Optional[str]
    Weather: Optional[str]
    Temp: Optional[float]
    ATemp: Optional[float]
    Humidity: Optional[float]
    Windspeed: Optional[float]
    Casual: Optional[int]
    Registered: Optional[int]
    Count: Optional[int]
    Year: Optional[str]
    Month: Optional[str]


class MultipleDataInputs(BaseModel):
    inputs: List[DataInputSchema]
