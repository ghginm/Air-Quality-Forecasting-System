import pandas as pd
import pyarrow.parquet as pq
from sqlalchemy import create_engine


## Loading data

def get_data(db_url: str, data_path: str, sql_access: bool = False,
             date_col: str = 'date', id_col: str = 'id',) -> pd.DataFrame:
    """Loading data.
    
    Parameters:
        db_url : SQL engine url.
        data_path : a path to locally stored data.
        sql_access : whether to load data from SQL or a local directory.
        date_col : the primary date column (day).
        id_col : a column that uniquely identifies the hierarchy of time series.
    """
    
    if sql_access:
        engine = create_engine(db_url)
        query = """
                select * from my_table
                """

        data = pd.read_sql(query, con=engine)
    else:
        with open(data_path, 'rb') as handle:
            data = pq.read_table(handle).to_pandas()

    # Sorting data (necessary to correctly create lag variables)
    data = data.sort_values(by=[id_col, date_col]).reset_index(drop=True)

    return data

##  Reducing memory usage

def squeeze_df(df: pd.DataFrame) -> pd.DataFrame:
    """Reduce memory usage.
    
    Parameters:
        df : pandas DataFrame with daily time series.
    """

    cols = dict(df.dtypes)

    for col, t in cols.items():
        if 'float' in str(t):
            df[col] = pd.to_numeric(df[col], downcast='float')
        if 'int' in str(t):
            df[col] = pd.to_numeric(df[col], downcast='integer')
        elif t == 'object':
            pass

    return df

## Reports

def continuous_features_report(df: pd.DataFrame, continuous_features: list) -> pd.DataFrame:
    """Creating a data report for continuous features.
    
    Parameters:
        df : pandas DataFrame with daily time series.
        continuous_features : a list of continuous features to analyse.
    """

    report = pd.DataFrame()

    for column in continuous_features:
        report.at[column, 'Mean'] = df[column].mean()
        report.at[column, 'Median'] = df[column].median()
        report.at[column, 'Std'] = df[column].std()
        report.at[column, 'N observations'] = df[column].count()
        report.at[column, 'Missing values %'] = (df[column].isnull().sum() / len(df))*100
        report.at[column, 'Cardinality'] = df[column].nunique()
        report.at[column, 'Min'] = df[column].min()
        report.at[column, '1st Quartile'] = df[column].quantile(0.25)
        report.at[column, '2nd Quartile'] = df[column].quantile(0.5)
        report.at[column, '3rd Quartile'] = df[column].quantile(0.75)
        report.at[column, 'Max'] = df[column].max()

    return report.round(2).sort_values(by='Missing values %', ascending=False)


def categorical_features_report(df: pd.DataFrame, categorical_features: list) -> pd.DataFrame:
    """Creating a data report for categorical features.
    
    Parameters:
        df : pandas DataFrame with daily time series.
        categorical_features : a list of categorical features to analyse.
    """
    
    report = pd.DataFrame()

    for column in categorical_features:
        report.at[column, 'N observations'] = df[column].count()
        report.at[column, 'Missing values %'] = (df[column].isnull().sum() / len(df))*100
        report.at[column, 'Cardinality'] = df[column].nunique()

    return report.round(2).sort_values(by='Missing values %', ascending=False)


if __name__ == '__main__':
    pass