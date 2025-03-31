import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

from titanic_model.config.core import config
from titanic_model.processing.features import embarkImputer
from titanic_model.processing.features import Mapper
from titanic_model.processing.features import age_col_tfr

titanic_pipe=Pipeline([
    
    ("embark_imputation", embarkImputer(variables=config.model_config.embarked_var)
     ),
     ##==========Mapper======##
     ("map_sex", Mapper(config.model_config.gender_var, config.model_config.gender_mappings)
      ),
     ("map_embarked", Mapper(config.model_config.embarked_var, config.model_config.embarked_mappings )
     ),
     ("map_title", Mapper(config.model_config.title_var, config.model_config.title_mappings)
     ),
     # Transformation of age column
     ("age_transform", age_col_tfr(config.model_config.age_var)
     ),
    # scale
     ("scaler", StandardScaler()),
     ('model_rf', RandomForestClassifier(n_estimators=config.model_config.n_estimators, max_depth=config.model_config.max_depth,
                                      random_state=config.model_config.random_state))
          
     ])