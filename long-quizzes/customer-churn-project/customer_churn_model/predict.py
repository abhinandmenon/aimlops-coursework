import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from typing import Union
import pandas as pd

from customer_churn_model import __version__ as _version
from customer_churn_model.config.core import config
from customer_churn_model.pipeline import customer_churn_pipe
from customer_churn_model.processing.data_manager import load_pipeline
from customer_churn_model.processing.validation import validate_inputs


pipeline_file_name = f"{config.app_config_.pipeline_save_file}{_version}.pkl"
customer_churn_pipe = load_pipeline(file_name=pipeline_file_name)


def make_prediction(*, input_data: Union[pd.DataFrame, dict]) -> dict:
    """Make a prediction using a saved model """

    validated_data, errors = validate_inputs(input_df=pd.DataFrame(input_data))
    
    results = {"predictions": None, "version": _version, "errors": errors}
    
    if not errors:
        results["predictions"] = customer_churn_pipe.predict(validated_data)
    
    print(results)
    return results

if __name__ == "__main__":

    data_in={'CustomerID':[2], 'Age':[30], 'Gender':['Female'], 'Tenure':[39], 'Usage Frequency':[14],
             'Support Calls':[5], 'Payment Delay':[18], 'Subscription Type':['Standard'], 'Contract Length':['Annual'],
             'Total Spend':[932.00], 'Last Interaction':[17]}

    make_prediction(input_data=data_in)
