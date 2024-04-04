import re
from itertools import chain, product

import numpy as np
import pandas as pd
from workalendar.usa.core import UnitedStates


## Calendar variables, e.g. holidays, weekdays etc.

def calendar_vars_daily_data(df: pd.DataFrame, date_col: str = 'date',
                             get_business_days: bool = True) -> pd.DataFrame:
    """Creating calendar variables.

    Parameters:
        df : pandas DataFrame with daily time series.
        date_col : the date column (day).
        get_business_days : creating a flag variable marking working days or not.
    """

    # Dataframe with a range of dates
    cal = UnitedStates()
    min_date = df[date_col].min()
    max_date = df[date_col].max()

    date_range = pd.date_range(start=min_date - pd.Timedelta(7, 'd'),
                               end=max_date + pd.Timedelta(7, 'd'), freq='d')

    df_days = pd.DataFrame({date_col: date_range})
    df_days['year'] = df_days[date_col].dt.year

    # Basic calendar variables
    df_days['date_day'] = df_days[date_col].dt.day
    df_days['date_week_year'] = [x.isocalendar()[1] for x in df_days[date_col]]
    df_days['date_month'] = df_days[date_col].dt.month
    df_days['date_weekday'] = df_days[date_col].dt.weekday + 1

    # Sine and cosine pairs
    for i in [1, 7, 28, 90]:
        df_days[f'f_sin_{i}'] = np.sin((2*i * np.pi * df_days['date_day']) / 365)
        df_days[f'f_cos_{i}'] = np.cos((2*i * np.pi * df_days['date_day']) / 365)

    # Basic holiday variables
    all_holidays = []

    for x in df_days['year'].unique():
        all_holidays.append(cal.holidays(x))

    all_holidays = pd.DataFrame(list(chain(*all_holidays)), columns=[date_col, 'holiday'])
    all_holidays[date_col] = pd.to_datetime(all_holidays[date_col])
    all_holidays['holiday'] = [re.sub('[^A-Za-z0-9_]+', '', x) for x in all_holidays['holiday']]

    # Marking working days
    if get_business_days:
        df_days['business_day'] = [int(cal.is_working_day(x)) for x in df_days[date_col]]

    # Merging calendar variables with the initial dataset
    df_days = df_days.drop('year', axis=1)
    df = df.merge(df_days, how='left', on=date_col)
    df = df.merge(all_holidays, how='left', on=date_col)
    df['holiday'] = df['holiday'].fillna('no')

    return df

## Lag variables (converting a time-series problem to the ML format)

# TODO: compute lags in parallel

def lag_vars(df: pd.DataFrame, date_col: str = 'date', id_col: str = 'id', target_col: str = 'target',
             n_lags: int = 10, distant_lags: list | None = None,  # [90, 180, 365], 
             window: list | None = [3, 7, 14, 28], window_lag: list | None = [1, 7, 14],
             suppress_warnings: bool = False) -> pd.DataFrame:
    """Creating lag variables.

    Parameters:
        df : pandas DataFrame with daily time series.
        date_col : the date column (day).
        id_col : a column that uniquely identifies the hierarchy of time series.
        target_col : the target variable.
        n_lags : the number of lag variables.
        distant_lags : specific lag variables.
        window : the size of the moving average window.
        window_lag : lags for shifting the moving average.
        suppress_warnings : a toggle to show / hide warnings.
    """

    # Warning
    if not suppress_warnings:
        print((f'* When concatenating DataFrame with lag variables, make sure that DataFrame '
               f'is sorted by "{id_col}" and "{date_col}" and that the index is reset.'))

    df_lags, col_names = [], []
    df = df.sort_values(by=[id_col, date_col]).reset_index(drop=True)
    df_group = df.groupby([id_col], observed=True, group_keys=False)[target_col]

    # Basic lags
    for i in range(1, n_lags + 1):
        col_name = f'lag_{i}'
        col_names.append(col_name)
        df_lags.append(pd.to_numeric(df_group.shift(i), downcast='float'))

    # Distant lags
    if distant_lags is not None:
        for i in distant_lags:
            col_name = f'lag_{i}'
            col_names.append(col_name)
            df_lags.append(pd.to_numeric(df_group.shift(i), downcast='float'))

    # Rolling window statistics (lagged)
    if (window is not None) and (window_lag is not None):
        # Rolling mean
        for i in product(window, window_lag):
            col_name = f'lag_{i[1]}_roll_{i[0]}_mean'
            col_names.append(col_name)
            df_lags.append(pd.to_numeric(df_group.shift(i[1]).rolling(i[0]).mean(), downcast='float'))

    df_lags = pd.DataFrame(df_lags, index=col_names).T.reset_index(drop=True)

    # Adding lag variables to the original dataset
    df = pd.concat([df, df_lags], axis=1).dropna().reset_index(drop=True)

    return df


def release_exogenous_vars(df: pd.DataFrame, date_col: str = 'date', id_col: str = 'id',
                           exogenous_cols: list = ['x1', 'x2', 'x3'],
                           n_release_vars: int = 7, suppress_warnings: bool = False) -> pd.DataFrame:
    """Creating release variables.

    Parameters:
        df : pandas DataFrame with daily time series.
        date_col : the date column (day).
        id_col : a column that uniquely identifies the hierarchy of time series.
        exogenous_cols : columns to calculate release features for.
        n_release_vars : the number of release features.
        suppress_warnings : a toggle to show / hide warnings.
    """

    # Warning
    if not suppress_warnings:
        print((f'* When concatenating DataFrame with lag variables, make sure that DataFrame '
               f'is sorted by "{id_col}" and "{date_col}" and that the index is reset.'))

    df_rel, col_names = [], []
    df = df.sort_values(by=[id_col, date_col]).reset_index(drop=True)
    df_group = df.groupby([id_col], observed=True, group_keys=False)[exogenous_cols]

    # Release features
    for i in range(1, n_release_vars + 1):
        df_rel.append(df_group.shift(i) / df[exogenous_cols])  # datatype is not efficient

    df_rel = pd.concat(df_rel, axis=1)

    # Feature names
    for i in range(1, n_release_vars + 1):
        for col in exogenous_cols:
            col_name = f'{col}_release_{i}'
            col_names.append(col_name)

    df_rel.columns = col_names

    # Replacing inf / -inf values that could happen during division
    df_rel = df_rel.replace([np.inf], np.nan)
    max_values = df_rel.max()
    df_rel = df_rel.fillna(max_values)

    df_rel = df_rel.replace([-np.inf], np.nan)
    min_values = df_rel.min()
    df_rel = df_rel.fillna(min_values)

    # Adding release variables to the original dataset
    df = pd.concat([df, df_rel], axis=1).reset_index(drop=True)

    return df


if __name__ == '__main__':
    pass