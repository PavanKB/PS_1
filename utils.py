from ucimlrepo import fetch_ucirepo 
from itertools import combinations
from sklearn.metrics import matthews_corrcoef
import pandas as pd
import numpy as np

def get_census_income():
    # fetch dataset 
    census_income = fetch_ucirepo(id=20) 
    
    # data (as pandas dataframes) 
    X = census_income.data.features 
    y = census_income.data.targets 
      
    # metadata 
    print(census_income.metadata) 
      
    # variable information 
    print(census_income.variables) 


def calc_matthews_corrcoef(X):
    """
    Calculate the pair wise matthews corr coeff for all the columns.

    returns a matrix 
    """
    corr_coeff = pd.DataFrame(np.nan, columns=X.columns, index=X.columns)
    
    pairs = list(combinations(X.columns, 2))

    for col in X.columns:
        corr_coeff.loc[col, col] = 1.0

    for col_x, col_y in combinations(X.columns, 2):

        corr_coeff.loc[col_x, col_y] = corr_coeff.loc[col_y, col_x] = matthews_corrcoef(X[col_x], X[col_y])

    return corr_coeff
        
    