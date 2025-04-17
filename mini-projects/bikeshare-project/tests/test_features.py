
"""
Note: These tests will fail if you have not first trained the model.
"""
import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import numpy as np
from bikeshare_model.config.core import config
from bikeshare_model.processing.features import WeekdayOneHotEncoder, UnusedColumnsDropper


def test_weekday_one_hot_encoder(sample_input_data):
    # Given
    input_data = sample_input_data[0]
    transformer = WeekdayOneHotEncoder(config.model_config_.weekday_var)
    
    assert config.model_config_.weekday_var in input_data
    assert input_data.loc[7091, config.model_config_.weekday_var] == 'Sun'

    # When
    encoded_data = transformer.fit(input_data).transform(input_data)

    # Then
    assert config.model_config_.weekday_var not in encoded_data
    assert encoded_data.loc[7091, config.model_config_.weekday_var + '_Sun'] == 1.0
    assert encoded_data.loc[7091, config.model_config_.weekday_var + '_Mon'] == 0.0
    assert encoded_data.loc[7091, config.model_config_.weekday_var + '_Tue'] == 0.0
    assert encoded_data.loc[7091, config.model_config_.weekday_var + '_Wed'] == 0.0
    assert encoded_data.loc[7091, config.model_config_.weekday_var + '_Thu'] == 0.0
    assert encoded_data.loc[7091, config.model_config_.weekday_var + '_Fri'] == 0.0
    assert encoded_data.loc[7091, config.model_config_.weekday_var + '_Sat'] == 0.0


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
