import preprocessing_functions as pf
import config
#from sklearn.model_selection import train_test_split
# ================================================
# TRAINING STEP - IMPORTANT TO PERPETUATE THE MODEL

# Load data
data = pf.load_data(config.PATH_TO_DATASET)

# divide data set
X_train, X_test, y_train, y_test = pf.divide_train_test(data,config.TARGET)

# get first letter from cabin variable
X_train['cabin'] = pf.extract_cabin_letter(X_train)

# impute categorical variables
for var in config.CATEGORICAL_VARS:
    X_train[var] = pf.impute_na(X_train,var)


# impute numerical variable
for var,value in config.IMPUTATION_DICT.items():
    X_train = pf.add_missing_indicator(X_train,var)
    X_train[var] = pf.impute_na(X_train,var,value)



# Group rare labels
X_train = pf.remove_rare_labels(X_train,config.FREQUENT_LABELS)

# encode categorical variables
X_train = pf.encode_categorical(X_train,config.CATEGORICAL_VARS)

# check all dummies were added
X_train = pf.check_dummy_variables(X_train,config.DUMMY_VARIABLES)


# train scaler and save
scaler = pf.train_scaler(X_train,config.OUTPUT_SCALER_PATH)

# scale train set
X_train = scaler.transform(X_train)


# train model and save
pf.train_model(X_train,y_train,config.OUTPUT_MODEL_PATH)

print('Finished training')