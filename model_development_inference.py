import json
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

import model_creation.training_testing as training_testing
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
config_split = config['train_test_split']

## Data

# Loading data
data = data_utils.get_data(db_url='mysql+mysqlconnector:...',
                           data_path=f'{path_project}\\data\\city_pollution_data.parquet',
                           sql_access=False, date_col=config_data['date_col'], id_col=config_data['id_col'])

# Splitting data
data_train, _, data_test = tuning_cv.train_val_test_split(df=data, date_col=config_data['date_col'], 
                                                          id_col=config_data['id_col'],
                                                          test_size=config_split['test_size'], val_size_es=None)

# Columns to encode
target_enc_cols, one_hot_cols = ['City'], ['holiday']

# Columns to ignore
ignore_cols = ['County', 'State']

# Model types
model_types = ['xgb', 'lgb', 'rf', 'lasso']

## Training

# Default models
tr_te = training_testing.TrainPredict(df_train=data_train, df_test=data_test, ignore_cols=ignore_cols,
                                      config=config, path=path_model,
                                      date_col=config_data['date_col'], id_col=config_data['id_col'], 
                                      target_col=config_data['target_col'],
                                      exogenous_cols=['wind-gust_median', 'temperature_max'],
                                      model_params_default=True)

for model_type in model_types:
    _ = tr_te.model_training(model_type=model_type, target_enc_cols=target_enc_cols, one_hot_cols=one_hot_cols)

# Tuned models
tr_te = training_testing.TrainPredict(df_train=data_train, df_test=data_test, ignore_cols=ignore_cols,
                                      config=config, path=path_model,
                                      date_col=config_data['date_col'], id_col=config_data['id_col'], 
                                      target_col=config_data['target_col'],
                                      exogenous_cols=['wind-gust_median', 'temperature_max'],
                                      model_params_default=False)

for model_type in model_types:
    _ = tr_te.model_training(model_type=model_type, target_enc_cols=target_enc_cols, one_hot_cols=one_hot_cols)

## Testing

fc_dates = data_test[config_data['date_col']].unique()
fc_dates_versions = [fc_dates[x:x+7] for x in range(0, len(fc_dates), 7)]

# Default models
fc_versions = []

for idx, version in enumerate(fc_dates_versions):
    data_train, _, data_test = tuning_cv.train_val_test_split(df=data, date_col=config_data['date_col'], 
                                                              id_col=config_data['id_col'],
                                                              test_size=None, val_size_es=None,
                                                              supply_test_dates=True, test_dates=version)

    tr_te = training_testing.TrainPredict(df_train=data_train, df_test=data_test, ignore_cols=ignore_cols,
                                          config=config, path=path_model,
                                          date_col=config_data['date_col'], id_col=config_data['id_col'], 
                                          target_col=config_data['target_col'],
                                          exogenous_cols=['wind-gust_median', 'temperature_max'],
                                          model_params_default=True)

    fc = tr_te.recursive_forecasting_multiple_models(model_type_list=model_types)
    fc['fc_version'] = idx
    fc_versions.append(fc)

fc_all_default = pd.concat(fc_versions)
fc_all_default['tuning'] = 'False'

# Tuned models
fc_versions = []

for idx, version in enumerate(fc_dates_versions):
    data_train, _, data_test = tuning_cv.train_val_test_split(df=data, date_col=config_data['date_col'], 
                                                              id_col=config_data['id_col'],
                                                              test_size=None, val_size_es=None,
                                                              supply_test_dates=True, test_dates=version)

    tr_te = training_testing.TrainPredict(df_train=data_train, df_test=data_test, ignore_cols=ignore_cols,
                                          config=config, path=path_model,
                                          date_col=config_data['date_col'], id_col=config_data['id_col'], 
                                          target_col=config_data['target_col'],
                                          exogenous_cols=['wind-gust_median', 'temperature_max'],
                                          model_params_default=False)

    fc = tr_te.recursive_forecasting_multiple_models(model_type_list=model_types)
    fc['fc_version'] = idx
    fc_versions.append(fc)

fc_all_tuned = pd.concat(fc_versions)
fc_all_tuned['tuning'] = 'True'

# SHAP (lgb)
fc_versions, fc_shap = [], []

for idx, version in enumerate(fc_dates_versions):
    data_train, _, data_test = tuning_cv.train_val_test_split(df=data, date_col=config_data['date_col'],
                                                              id_col=config_data['id_col'],
                                                              test_size=None, val_size_es=None,
                                                              supply_test_dates=True, test_dates=version)

    tr_te = training_testing.TrainPredict(df_train=data_train, df_test=data_test, ignore_cols=ignore_cols,
                                          config=config, path=path_model,
                                          date_col=config_data['date_col'], id_col=config_data['id_col'],
                                          target_col=config_data['target_col'],
                                          exogenous_cols=['wind-gust_median', 'temperature_max'],
                                          model_params_default=False)

    _, df_shap = tr_te.recursive_forecasting(model_type='lgb', generate_shap=True)
    df_shap['fc_version'] = idx
    fc_shap.append(df_shap)

shap_all_lgb = pd.concat(fc_shap)

# Storing results
fc_all = pd.concat([fc_all_default, fc_all_tuned])
fc_all = fc_all.rename(columns={config_data['target_col']: f"{config_data['target_col']}_forecast"})

fc_all = fc_all.merge(data[[config_data['date_col'], config_data['id_col'], config_data['target_col']]],
                      how='left', on=[config_data['date_col'], config_data['id_col']])

with open(f'{path_project}\\data\\forecast.parquet', 'wb') as handle:
    pq.write_table(pa.Table.from_pandas(fc_all), handle, compression='GZIP')

with open(f'{path_project}\\data\\shap_lgb.parquet', 'wb') as handle:
    pq.write_table(pa.Table.from_pandas(shap_all_lgb), handle, compression='GZIP')