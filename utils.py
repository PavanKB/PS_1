from matplotlib import pyplot as plt
from ucimlrepo import fetch_ucirepo 
from itertools import combinations, product
from sklearn.metrics import matthews_corrcoef
import pandas as pd
import numpy as np
import seaborn as sns


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
    

def get_df_details(df):
    """Get them details of the dataframe column wise, with more info than df.describe()

    Args:
        df (pd.DataFrame): The dataframe in wide format to be analysed

    Returns:
        pd.DataFrame: Summary of the dataframe
    """
    df_summ = df.describe(include='all').transpose()
    df_summ['nunique'] = df.nunique()
    df_summ['n_nulls'] = df.isnull().sum()
    df_summ['dtype'] = df.dtypes

    return df_summ
    

def plot_distrb(df):
    """
    plot value counts for each column as subplots
    if the column is categorical, use countplot
    if the column is numerical, use distplot

    # https://www.statology.org/seaborn-subplots/

    Args:
        df (pd.DataFrame): DataFrame in wide formatr to plot
    """
    plot_rows = np.ceil(np.sqrt(df.shape[1])).astype(int)

    fig, ax = plt.subplots(plot_rows, plot_rows, figsize=(20, 20))

    for i in product(range(plot_rows), range(plot_rows)):

        idx = (i[0]+1) * (i[1]+1)
        if idx >= df.shape[1]:
            break

        col = df.iloc[:, idx]
        if col.dtype == 'object':
            sns.countplot(col, ax=ax[i])
        else:
            sns.histplot(col, ax=ax[i])
