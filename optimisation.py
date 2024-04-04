import json
from pathlib import Path

import pandas as pd
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

import model_creation.feature_engineering as feature_engineering
import model_creation.tuning_cv as tuning_cv
import utils.data_utils as data_utils


## Initial parameters

# Paths
path_project = str(Path('__file__').absolute().parent)
path_model = f'{path_project}\\model'

# Config
with open(f'{path_project}\\config.json', 'r') as config_file:
    config = json.load(config_file)

config_data = config['dataset']
config_features = config['features']
config_split = config['train_test_split']
config_cv = config['cv']
config_tuning = config['tuning']

## Data

# Loading data
data = data_utils.get_data(db_url='mysql+mysqlconnector:...',
                           data_path=f'{path_project}\\data\\city_pollution_data.parquet',
                           sql_access=False, date_col=config_data['date_col'], id_col=config_data['id_col'])

# Features
data = feature_engineering.calendar_vars_daily_data(df=data, date_col=config_data['date_col'], get_business_days=True)

data = feature_engineering.release_exogenous_vars(df=data, date_col=config_data['date_col'], id_col=config_data['id_col'],
                                                  exogenous_cols=['wind-gust_median', 'temperature_max'], n_release_vars=7)

data = feature_engineering.lag_vars(df=data, date_col=config_data['date_col'], id_col=config_data['id_col'],
                                    target_col=config_data['target_col'], n_lags=config_features['n_lags'],
                                    window=config_features['window'],
                                    window_lag=config_features['window_lag'])

# Splitting data
data_train, data_val, _ = tuning_cv.train_val_test_split(df=data, date_col=config_data['date_col'],
                                                         id_col=config_data['id_col'],
                                                         test_size=config_split['test_size'],
                                                         val_size_es=config_split['val_size_es'])

keep_cols = list(set(data.columns) - set(['Date', 'County', 'State'] + [config_data['target_col']]))
X_train, y_train = data_train[keep_cols], data_train[config_data['target_col']]
X_eval, y_eval = data_val[keep_cols], data_val[config_data['target_col']]

del data

# Columns to encode
target_enc_cols, one_hot_cols = ['City'], ['holiday']

## Custom cross-validation

cv = tuning_cv.custom_cv(df=data_train, date_col=config_data['date_col'],
                         val_size=config_cv['val_size'], n_splits=config_cv['n_splits'])

del data_train, data_val

## Tuning models

print('* Baseline XGB with default hyperparameters')
xgb_default_scores = tuning_cv.cv_early_stopping(X_train=X_train, y_train=y_train,
                                                 X_eval=X_eval, y_eval=y_eval,
                                                 model=XGBRegressor(), cv=cv, target_enc_cols=target_enc_cols,
                                                 one_hot_cols=one_hot_cols, baseline_model=True, parallelism='both')

print('* Optuna XGB')
xgb_scores = tuning_cv.optuna_tuning(X_train, y_train, X_eval, y_eval, path=path_model,
                                     model_type='xgb', cv=cv, early_stopping_rounds=config_tuning['early_stopping_rounds'],
                                     target_enc_cols=target_enc_cols, one_hot_cols=one_hot_cols,
                                     parallelism='both', optuna_n_trials=config_tuning['optuna_n_trials'])

print('* Baseline LGB with default hyperparameters')
lgb_default_scores = tuning_cv.cv_early_stopping(X_train=X_train, y_train=y_train,
                                                 X_eval=X_eval, y_eval=y_eval,
                                                 model=LGBMRegressor(), cv=cv, target_enc_cols=target_enc_cols,
                                                 one_hot_cols=one_hot_cols, baseline_model=True, parallelism='both')

print('* Optuna LGB')
lgb_scores = tuning_cv.optuna_tuning(X_train, y_train, X_eval, y_eval, path=path_model,
                                     model_type='lgb', cv=cv, early_stopping_rounds=config_tuning['early_stopping_rounds'],
                                     target_enc_cols=target_enc_cols, one_hot_cols=one_hot_cols,
                                     parallelism='both', optuna_n_trials=config_tuning['optuna_n_trials'])

print('* Random search RF')
rf_scores = tuning_cv.random_search_tuning(X_train, y_train, path=path_model,
                                           model_type='rf', cv=cv,
                                           target_enc_cols=target_enc_cols, one_hot_cols=one_hot_cols,
                                           parallelism='both', n_iter=config_tuning['n_iter'])

print('* Random search Lasso')
lasso_scores = tuning_cv.random_search_tuning(X_train, y_train, path=path_model,
                                              model_type='lasso', cv=cv,
                                              target_enc_cols=target_enc_cols, one_hot_cols=one_hot_cols,
                                              parallelism='both', n_iter=config_tuning['n_iter'])

# Saving CV results
cv_results = pd.DataFrame([xgb_default_scores[0:2], xgb_scores,
                           lgb_default_scores[0:2], lgb_scores, rf_scores, lasso_scores],
                          columns=['rmse_avg', 'rmse_std'],
                          index=['xgb_default', 'xgb_optuna', 'lgb_default', 'lgb_optuna',
                                 'rf_random_search', 'lasso_random_search'])

cv_results.to_excel(f'{path_model}\\cv_results.xlsx', index=True)