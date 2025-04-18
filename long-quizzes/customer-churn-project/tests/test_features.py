
"""
Note: These tests will fail if you have not first trained the model.
"""
import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import numpy as np
from customer_churn_model.config.core import config
from customer_churn_model.processing.features import RatioFeaturesComputer, CustomOneHotEncoder, UnusedColumnsDropper


def test_ratio_features_computer(sample_input_data):
    # Given
    input_data = sample_input_data[0]
    transformer = RatioFeaturesComputer(config.model_config_.ratio_features)
    print(input_data[1000:1010])
    
    for num_var, denom_var in config.model_config_.ratio_features.items():
        assert num_var in input_data
        assert denom_var in input_data

    # When
    encoded_data = transformer.fit(input_data).transform(input_data)

    # Then
    for num_var, denom_var in config.model_config_.ratio_features.items():
        assert num_var + ' by ' + denom_var in encoded_data


def test_custom_one_hot_encoder(sample_input_data):
    # Given
    input_data = sample_input_data[0]
    transformer = CustomOneHotEncoder(config.model_config_.categorical_features)    

    for var in config.model_config_.categorical_features:
        assert var in input_data

    # When
    encoded_data = transformer.fit(input_data).transform(input_data)

    # Then
    for var in config.model_config_.categorical_features:
        assert var not in encoded_data


def test_unused_columns_dropper(sample_input_data):
    # Given
    input_data = sample_input_data[0]
    transformer = UnusedColumnsDropper(config.model_config_.unused_columns)
    
    for column in config.model_config_.unused_columns:
        assert column in input_data

    # When
    encoded_data = transformer.fit(input_data).transform(input_data)

    # Then
    for column in config.model_config_.unused_columns:
        assert column not in encoded_data
