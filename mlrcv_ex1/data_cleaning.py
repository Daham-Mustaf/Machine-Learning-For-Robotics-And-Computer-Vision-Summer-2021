import numpy as np
import pandas as pd
import typing

def get_columns_with_missing_values(df):
    na_columns = df.columns[df.isna().any()].tolist()
    return na_columns


def replace_nan_with_mean_class(df: pd.DataFrame, col: str, refcol: str) -> pd.DataFrame:
    """
    This function should go over all the column (col) values where the it is NaN, then for each row:
    - Get the refcol value
    - Calculate the mean value of col for the refcol class
    - Replace the row nan value with the respective mean refcol value

    Args:
        - df (pd.DataFrame): dataframe to replace nans with mean class
        - col (str): column to replace the NaNs
        - refcol (str): column to be used as reference to calculate the mean value of the col value

    Returns:
        - df (pd.DataFrame): dataframe with the mass NaNs replaced with mean mass of the recclass
    """
    df[col] = df.groupby(refcol)[col].transform(lambda x: x.fillna(x.mean()))

    return df

def categorical_to_num(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    This function should list all the categories within the given column (col) and
    assign a category id for each individual category, then for each row:
    - Replace the category value with the respective category id number

    Args:
        - df (pd.DataFrame): dataframe to change the categorical data to numerical
        - col (str): column to be changed

    Returns:
        - df (pd.DataFrame): dataframe with the categorical class replaced with categories ids numbers
    """
    pass

    return df

def remove_nan_rows(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    This function should list all the nan values within the given column then for each row:
    - Remove the current row if the column (col) is NaN

    Args:
        - df (pd.DataFrame): dataframe to have the NaNs drop
        - col (str): column in which the NaNs will be removed

    Returns:
        - df (pd.DataFrame): dataframe without the rows where the column (col) contains NaN
    """
    pass

    return df

def remove_row_within_range(df: pd.DataFrame, col: str, min_val: float, max_val: float) -> pd.DataFrame:
    """
    This function should remove rows where the column (col) value is not inside a given range (min_val, max_val).
    For each row:
    - Check if the column (col) attribute is inside the range:
        - if is not in range: drop the row
        - if is in range: keep the row

    Args:
        - df (pd.DataFrame): dataframe that will have the rows removed
        - col (str): column to check the value (and decide if the row will be removed)
        - min_val (float): min allowed value for the column (values below this will be removed)
        - max_val (float): max allowed value for the column (values above this will be removed)

    Returns:
        - df (pd.DataFrame): dataframe without the rows where column (col) is not within range (min_val, max_val)
    """
    pass

    return df

def remap_values(df: pd.DataFrame, col: str, remap_dict: dict) -> pd.DataFrame:
    """
    This function should remap all values from a given column (col) using a remap_dict (should be defined first).
    For each row:
    - Replace the column (col) value the mapped value on dict (rempap_dict)
    Args:
        - df (pd.DataFrame): dataframe to have the values remapped
        - col (str): column that the values will be remapped
        - remap_dict (dict): dictionary mapping the changes ('old_value': 'new_value')

    Returns:
        - df (pd.DataFrame): dataframe with all the values from column (col) remapped according to remap_dict
    """
    pass

    return df

def drop_column(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    This function drop a given column (col). For each row:
    - Drop values from the column (col)

    Args:
        - df (pd.DataFrame): dataframe to have the columns removed
        - col (str): column to be removed

    Returns:
        - df (pd.DataFrame): dataframe without the given column
    """
    pass

    return df

# Dataset read
df = pd.read_csv('meteorite-landings.csv')

columns_with_missing_values = get_columns_with_missing_values(df)
print(columns_with_missing_values)

df.groupby(df['recclass'])['mass'].transform(lambda x: x.fillna(x.mean()))
df['mass'].isna().sum()
columns_with_missing_values = get_columns_with_missing_values(df)
print(columns_with_missing_values)

replace_nan_with_mean_class(df, 'year', 'recclass')
replace_nan_with_mean_class(df, 'reclat', 'recclass')
df['reclat'].isna().sum()
categories, _ = pd.factorize(df['reclat'])



df.info()
df['recclass'].unique()
df.columns
# number of missing values (NaN) in each column of the DataFrame.
df.isna().sum()
df.groupby('recclass') 


df.columns

na_values = df['name', 'id', 'nametype', 'recclass', 'mass', 'fall', 'year', 'reclat',
       'reclong', 'GeoLocation']

na_values = df[ df['mass'].isna() & df['year'].isna() & df['reclat'].isna() & df['reclong'].isna()  & df['GeoLocation'].isna()]
na_values.shape
na_values.hist(bins=60, figsize=(20,15))

na_columns = df.columns[df.isna().any()].tolist()

import pandas as pd
import matplotlib.pyplot as plt

def plot_missing_data_histograms(data):
    # Get the columns with missing data
    na_columns = data.columns[data.isna().any()].tolist()
    
    na_columns.hist(bins=60, figsize=(20,15))


def plot_missing_histograms(df, na_columns):
    for col in na_columns:
        na_values = df[df[col].isna()]
        na_values.hist(bins=60, figsize=(20,15))
        plt.title(f'{col} missing values')
        plt.show()

na_columns = df.columns[df.isna().any()].tolist()
plot_missing_histograms(df, na_columns)

plot_missing_data_histograms(df)


df['mass'].isnull
df.duplicated()
df.isna().sum()
print(df[df['mass'].isna()])
df['GeoLocation'].unique()
df.groupby('GeoLocation')
df = replace_nan_with_mean_class(df, 'GeoLocation', 'GeoLocation')


df.isnull().values.any()
