import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

import joblib


# Individual pre-processing and training functions
# ================================================

def load_data(df_path):
    # Function loads data for training
    return pd.read_csv(df_path)



def divide_train_test(df, target):
    # Function divides data set in train and test
    X_train, X_test, y_train, y_test = train_test_split(df.drop(target,axis=1),df[target],test_size=0.2, random_state=0)
    return X_train, X_test, y_train, y_test 
    



def extract_cabin_letter(df):
    # captures the first letter
    return df['cabin'].str[0] 
    

def add_missing_indicator(df, var):
    # function adds a binary missing value indicator
    df[var+'_NA'] = np.where(df[var].isnull(), 1, 0)
    return df

    
def impute_na(df,var,value='Missing'):
    # function replaces NA by value entered by user
    # or by string Missing (default behaviour)
    return df[var].fillna(value)



def remove_rare_labels(df, frequent_labels):
    # groups labels that are not in the frequent list into the umbrella
    # group Rare
    for var,frq_list in frequent_labels.items():
        df[var] = np.where(df[var].isin(frq_list), df[var], 'Rare')
            
    return df



def encode_categorical(df, vars_cat):
    # adds ohe variables and removes original categorical variable
    for var in vars_cat:
            df = pd.concat([df, pd.get_dummies(df[var], prefix=var, drop_first=True) ], axis=1)

    df.drop(labels=vars_cat, axis=1, inplace=True)
    return df



def check_dummy_variables(df, dummy_list):
    
    # check that all missing variables where added when encoding, otherwise
    # add the ones that are missing
    colmuns = [c for c in dummy_list if c not in df.columns]
    for col in colmuns:
        df[col] = 0
    return df
    
    

def train_scaler(df, output_path):
    # train and save scaler
    scaler = StandardScaler()

    #  fit  the scaler to the train set
    scaler.fit(df) 
    joblib.dump(scaler,output_path)
    return scaler  
    

def scale_features(df, output_path):
    # load scaler and transform data
    scaler = joblib.load(output_path)
    return scaler.transform(df)



def train_model(X_train, y_train, output_path):
    # train and save model
    model = LogisticRegression(C=0.0005, random_state=0)

    # train the model
    model.fit(X_train, y_train)
    joblib.dump(model,output_path)
    return model


def predict(df, output_path):
    # load model and get predictions
    model = joblib.load(output_path)
    return model.predict(df)
