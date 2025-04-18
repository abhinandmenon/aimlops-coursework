import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from customer_churn_model.config.core import config
from customer_churn_model.pipeline import customer_churn_pipe
from customer_churn_model.processing.data_manager import load_dataset, save_pipeline

def run_training() -> None:
    """
    Train the model.
    """

    # read training data
    data = load_dataset(file_name=config.app_config_.training_data_file)
    X_train, y_train = data[config.model_config_.features], data[config.model_config_.target]

    # fit pipeline
    customer_churn_pipe.fit(X_train, y_train)

    # persist trained model
    save_pipeline(pipeline_to_persist= customer_churn_pipe)
    
if __name__ == "__main__":
    run_training()
