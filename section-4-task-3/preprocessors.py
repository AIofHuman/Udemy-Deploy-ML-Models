import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin


# Add binary variable to indicate missing values
class MissingIndicator(BaseEstimator, TransformerMixin):

    def __init__(self, variables=None):
        if not isinstance(variables,list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self, X, y=None):
        # to accommodate sklearn pipeline functionality
        return self

    def transform(self, X):
        # add indicator
        X = X.copy()
        for var in self.variables:
            # add missing indicator
            X[var+'_NA'] = np.where(X[var].isnull(), 1, 0)
        return X


# categorical missing value imputer
class CategoricalImputer(BaseEstimator, TransformerMixin):

    def __init__(self, variables=None):
        if not isinstance(variables,list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self, X, y=None):
        # we need the fit statement to accommodate the sklearn pipeline
        return self

    def transform(self, X):
        X = X.copy()
        for var in self.variables:
           X[var] = X[var].fillna('Missing') 
        return X


# Numerical missing value imputer
class NumericalImputer(BaseEstimator, TransformerMixin):

    def __init__(self, variables=None):
        if not isinstance(variables,list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self, X, y=None):
        # persist mode in a dictionary
        self.imputer_dict_ = {}
        for var in self.variables:
            self.imputer_dict_[var] = X[var].median()
        return self

    def transform(self, X):

        X = X.copy()
        for var in self.variables:
            X[var] = X[var].fillna(self.imputer_dict_[var])
        return X


# Extract first letter from string variable
class ExtractFirstLetter(BaseEstimator, TransformerMixin):

    def __init__(self, variables=None):
        if not isinstance(variables,list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self, X, y=None):
        # we need this step to fit the sklearn pipeline
        return self

    def transform(self, X):
        X = X.copy()
        for var in self.variables:
            X[var] = X[var].str[0]
        return X

# frequent label categorical encoder
class RareLabelCategoricalEncoder(BaseEstimator, TransformerMixin):

    def __init__(self, tol=0.05, variables=None):
        self.tol = tol
        if not isinstance(variables,list):
            self.variables = [variables]
        else:
            self.variables = variables

    def find_frequent_labels(self,df,var):
        '''
        Function finds the labels that are shared by more than
        a certain % of the rows in the dataset
        '''
        tmp = df[var].value_counts() / df.shape[0]
        return tmp[tmp > self.tol].index


    def fit(self, X, y=None):

        # persist frequent labels in dictionary
        self.encoder_dict_ = {}
        
        for var in self.variables:
            frequent_ls = self.find_frequent_labels(X, var)
            self.encoder_dict_[var] = frequent_ls
        return self

    def transform(self, X):
        X = X.copy()
        for var in self.variables:
            X[var] = np.where(X[var].isin(self.encoder_dict_[var]), X[var], 'Rare')
        return X

# string to numbers categorical encoder
class CategoricalEncoder(BaseEstimator, TransformerMixin):

    def __init__(self, variables=None):
        if not isinstance(variables,list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self, X, y=None):

        # HINT: persist the dummy variables found in train set
        self.dummies = pd.get_dummies(X[self.variables], drop_first=True).columns
        
        return self

    def transform(self, X):
        # encode labels
        X = X.copy()
        for var in self.variables:
            # get dummies
            X = pd.concat([X, pd.get_dummies(X[var], prefix=var, drop_first=True)], axis=1)
           
        # drop original variables
        X = X.drop(labels=self.variables, axis=1)
        # add missing dummies if any
        colmuns = [c for c in self.dummies if c not in X.columns]
        for col in colmuns:
            X[col] = 0

        return X
