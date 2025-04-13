import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

from bikeshare_model.config.core import config
from bikeshare_model.processing.features import WeekdayImputer, WeatherImputer
from bikeshare_model.processing.features import Mapper
from bikeshare_model.processing.features import OutlierHandler, WeekdayOneHotEncoder, UnusedColumnsDropper

bikeshare_pipe = Pipeline([

    ##========== Impute missing values ======##
    ('impute_missing_weekday', WeekdayImputer(config.model_config_.date_var, config.model_config_.weekday_var)),
    ('impute_missing_weather', WeatherImputer(config.model_config_.weather_var)),

    ##========== Map ordinal variables ======##
    ('map_year', Mapper(config.model_config_.year_var, config.model_config_.year_mappings)),
    ('map_month', Mapper(config.model_config_.month_var, config.model_config_.month_mappings)),
    ('map_season', Mapper(config.model_config_.season_var, config.model_config_.season_mappings)),
    ('map_weather', Mapper(config.model_config_.weather_var, config.model_config_.weather_mappings)),
    ('map_holiday', Mapper(config.model_config_.holiday_var, config.model_config_.holiday_mappings)),
    ('map_workingday', Mapper(config.model_config_.workingday_var, config.model_config_.workingday_mappings)),
    ('map_hour', Mapper(config.model_config_.hour_var, config.model_config_.hour_mappings)),

    ##========== Handle outliers in numerical variables ======##
    ('handle_outliers_temp', OutlierHandler(config.model_config_.temp_var)),
    ('handle_outliers_atemp', OutlierHandler(config.model_config_.atemp_var)),
    ('handle_outliers_humidity', OutlierHandler(config.model_config_.humidity_var)),
    ('handle_outliers_windspeed', OutlierHandler(config.model_config_.windspeed_var)),

    ##========== One-hot encode weekday variable ======##
    ('onehot_encode_weekday', WeekdayOneHotEncoder(config.model_config_.weekday_var)),

    ##========== Drop unused columns ======##
    ('drop_unused_columns', UnusedColumnsDropper(config.model_config_.unused_columns)),

    ##========== Scale ======##
    ('scale', StandardScaler()),

    ##========== Fit RF model ======##
    ('fit_rf_model', RandomForestRegressor(n_estimators=config.model_config_.n_estimators,
                                           max_depth=config.model_config_.max_depth,
                                           random_state=config.model_config_.random_state))
])
