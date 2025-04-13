import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from bikeshare_model.config.core import config
from bikeshare_model.pipeline import bikeshare_pipe
from bikeshare_model.processing.data_manager import load_dataset, save_pipeline

def run_training() -> None:
    """
    Train the model.
    """

    # read training data
    data = load_dataset(file_name=config.app_config_.training_data_file)
    X_train, y_train = data[config.model_config_.features], data[config.model_config_.target]

    # fit pipeline
    bikeshare_pipe.fit(X_train, y_train)

    # persist trained model
    save_pipeline(pipeline_to_persist= bikeshare_pipe)
    
if __name__ == "__main__":
    run_training()
