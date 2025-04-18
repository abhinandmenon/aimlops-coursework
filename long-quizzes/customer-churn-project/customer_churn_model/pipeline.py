import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

from customer_churn_model.config.core import config
from customer_churn_model.processing.features import RatioFeaturesComputer, CustomOneHotEncoder, UnusedColumnsDropper

customer_churn_pipe = Pipeline([

    ##========== Compute additional numerical features ======##
    ('compute_ratio_features', RatioFeaturesComputer(config.model_config_.ratio_features)),

    ##========== One hot encode categorical variables ======##
    ('one_hot_encode', CustomOneHotEncoder(config.model_config_.categorical_features)),

    ##========== Drop unused columns ======##
    ('drop_unused_columns', UnusedColumnsDropper(config.model_config_.unused_columns)),

    ##========== Scale ======##
    ('scale', StandardScaler()),

    ##========== Fit RF model ======##
    ('fit_rf_model', RandomForestClassifier(n_estimators=config.model_config_.n_estimators,
                                            max_depth=config.model_config_.max_depth,
                                            random_state=config.model_config_.random_state))

])
