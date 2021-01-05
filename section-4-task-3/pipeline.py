from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import preprocessors as pp
import config


titanic_pipe = Pipeline( 
    [
            # complete with the list of steps from the preprocessors file
            # and the list of variables from the config
            ('extract_first_letter',
                    pp.ExtractFirstLetter(variables=config.CABIN)),
            ('add_missing_indicator',
                    pp.MissingIndicator(variables=config.NUMERICAL_VARS)),
            ('fillna_median_numerical',
                    pp.NumericalImputer(variables=config.NUMERICAL_VARS)),
            ('fillna_missing_categorical',
                    pp.CategoricalImputer(variables=config.CATEGORICAL_VARS)),
            ('remove_labels',
                    pp.RareLabelCategoricalEncoder(config.TOLLERANCE,variables=config.CATEGORICAL_VARS)),
            ('one_hot_encoding',
                    pp.CategoricalEncoder(variables=config.CATEGORICAL_VARS)),
            ('scaler', StandardScaler()),
            ('Linear_model', LogisticRegression(C=0.0005, random_state=0))
            
    ]
)