import pickle
from typing import Literal

import numpy as np
import pandas as pd
import shap
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import TargetEncoder, OneHotEncoder
from xgboost import XGBRegressor

import model_creation.feature_engineering as feature_engineering


class TrainPredict:
    """This class is used for training and testing models. Since recursive forcasting is used, after each
    one-step-ahead forecast both the dataset and features have to be updated.

    Parameters:
        df_train : pandas DataFrame with daily time series (train set).
        df_test : pandas DataFrame with daily time series (test set).
        ignore_cols : columns that will be ignored during model fitting.
        config : a config json file.
        path : a directory where to store final parameters.
        date_col : the date column (day).
        id_col : a column that uniquely identifies the hierarchy of time series.
        target_col : the target variable.
        exogenous_cols : columns to calculate release features for.
        model_params_default : a flag variable to choose between default and tuned models.
    """

    def __init__(self, df_train: pd.DataFrame, df_test: pd.DataFrame, ignore_cols: list, config: dict, path: str,
                 date_col: str = 'date', id_col: str = 'id', target_col: str = 'target',
                 exogenous_cols: list = ['x1', 'x2', 'x3'], model_params_default: bool = False):
        self.df_train = df_train.copy()
        self.df_test = df_test.copy()
        self.ignore_cols = ignore_cols
        self.config = config
        self.path = path
        self.date_col = date_col
        self.id_col = id_col
        self.exogenous_cols = exogenous_cols
        self.target_col = target_col
        self.model_params_default = model_params_default

        if self.model_params_default:
            self.tuning = 'default'
        else:
            self.tuning = 'tuned'

        self.fc_periods = self.df_test[self.date_col].unique()

    def model_training(self, model_type: Literal['xgb', 'lgb', 'rf', 'lasso'],
                       target_enc_cols: list | None, one_hot_cols: list | None):
        """Training models. Two options are available: training models with default hyperparameters and
        training models with pre-tuned hyperparameters.

        Parameters:
            model_type : a type of sklearn ML algorithm.
            target_enc_cols : variables that should be target-encoded.
            one_hot_cols : variables that should be one-hot-encoded.
        """

        # Features (df_train exists within the scope of the function as compared to self.df_train)
        config_features = self.config['features']

        df_train = feature_engineering.calendar_vars_daily_data(df=self.df_train, date_col=self.date_col, get_business_days=True)

        df_train = feature_engineering.release_exogenous_vars(df=df_train, date_col=self.date_col, id_col=self.id_col,
                                                              exogenous_cols=self.exogenous_cols, n_release_vars=7)

        df_train = feature_engineering.lag_vars(df=df_train, date_col=self.date_col, id_col=self.id_col,
                                                target_col=self.target_col,
                                                n_lags=config_features['n_lags'],
                                                distant_lags=config_features['distant_lags'],
                                                window=config_features['window'],
                                                window_lag=config_features['window_lag'])

        keep_cols = list(set(df_train.columns) - set([self.date_col, self.target_col] + self.ignore_cols))

        # Target variable
        y_train = df_train[self.target_col].to_numpy()

        # ColumnTransformer
        transformers = []

        if target_enc_cols is not None:
            transformers.append(('target_encoding', TargetEncoder(target_type='continuous'), target_enc_cols))
        if one_hot_cols is not None:
            transformers.append(('one_hot_encoding', OneHotEncoder(handle_unknown='ignore'), one_hot_cols))

        col_transform = ColumnTransformer(
            transformers=transformers, remainder='passthrough', verbose_feature_names_out=False, n_jobs=-1
        )

        df_train = col_transform.fit_transform(df_train[keep_cols], y_train)
        df_train = pd.DataFrame(df_train, columns=col_transform.get_feature_names_out())

        with open(f'{self.path}\\col_transform.pickle', 'wb') as handle:
            pickle.dump(col_transform, handle)

        # Scaling data
        col_scale = StandardScaler()

        df_train = col_scale.fit_transform(df_train)
        df_train = pd.DataFrame(df_train, columns=col_scale.get_feature_names_out())

        with open(f'{self.path}\\col_scale.pickle', 'wb') as handle:
            pickle.dump(col_scale, handle)

        # Setting up hyperparameters
        if self.model_params_default:
            model_params = {}
        else:
            with open(f'{self.path}\\params_final_{model_type}.pickle', 'rb') as handle:
                model_params = pickle.load(handle)

        # Training a model
        if model_type == 'xgb':
            model = XGBRegressor()
            model.set_params(**{'n_jobs': -1})
        elif model_type == 'lgb':
            model = LGBMRegressor()
            model.set_params(**{'n_jobs': -1, 'verbose': -1})
        elif model_type == 'rf':
            model = RandomForestRegressor()
            model.set_params(**{'n_jobs': -1})
        elif model_type == 'lasso':
            model = Lasso(max_iter=2000)
        else:
            raise ValueError('The `model_type` parameter should be either "xgb", "lgb", "rf" or "lasso".')

        model.set_params(**model_params)
        model.fit(df_train, y_train)  # sample_weight=weights

        with open(f'{self.path}\\model_{model_type}_{self.tuning}.pickle', 'wb') as handle:
            pickle.dump(model, handle)

        return model, col_transform, col_scale


    def recursive_forecasting(self, model_type: Literal['xgb', 'lgb', 'rf', 'lasso'],
                              generate_shap: bool = False):
        """Employing the recursive forecasting strategy: one-step-ahead forecast is made,
        the output is then taken to recalculate lag variables.

        Parameters:
            model_type : a type of sklearn ML algorithm.
        """

        # Storing a subset of train data to properly create lag variables (df_test exists within the scope of the function as compared to self.df_test)
        config_features = self.config['features']

        window = np.max(config_features['window'])
        window_lag = np.max(config_features['window_lag'])
        max_window_lag = np.max([0 if window is None else window] + [0 if window_lag is None else window_lag])
        max_lag = config_features['n_lags']
        max_distant_lag = np.max(config_features['distant_lags'])
    
        if max_distant_lag is None:
            max_distant_lag = 0

        days_to_add = np.max([max_window_lag, max_lag, max_distant_lag]) + 1

        df_train_sub = self.df_train[self.df_train[self.date_col].isin(self.df_train[self.date_col].unique()[-days_to_add:])]

        self.df_test[self.target_col] = 0
        df_test = pd.concat([df_train_sub, self.df_test], axis=0).sort_values(by=[self.id_col, self.date_col]).reset_index(drop=True)

        # A placeholder for the number of the day in a given forecast version
        df_test['day_number'] = 0

        # Loading a model
        if model_type in ['xgb', 'lgb', 'rf', 'lasso']:
            with open(f'{self.path}\\model_{model_type}_{self.tuning}.pickle', 'rb') as handle:
                model = pickle.load(handle)
        else:
            raise ValueError('The `model_type` parameter should be either "xgb", "lgb", "rf" or "lasso".')

        # Loading feature scalers and transformers
        with open(f'{self.path}\\col_transform.pickle', 'rb') as handle:
            col_transform = pickle.load(handle)

        with open(f'{self.path}\\col_scale.pickle', 'rb') as handle:
            col_scale = pickle.load(handle)

        # Features
        df_test = feature_engineering.calendar_vars_daily_data(df=df_test, date_col=self.date_col, get_business_days=True)

        df_test = feature_engineering.release_exogenous_vars(df=df_test, date_col=self.date_col, id_col=self.id_col,
                                                             exogenous_cols=self.exogenous_cols, n_release_vars=7,
                                                             suppress_warnings=True)
        # SHAP values
        fc_shap, structure_shap = [], []

        # Recursive forecasting
        for idx, date_fc in enumerate(self.fc_periods):
            # Updating lag variables
            data_fc_iter = feature_engineering.lag_vars(df=df_test, date_col=self.date_col, id_col=self.id_col,
                                                        target_col=self.target_col,
                                                        n_lags=config_features['n_lags'],
                                                        distant_lags=config_features['distant_lags'],
                                                        window=config_features['window'],
                                                        window_lag=config_features['window_lag'], suppress_warnings=True)

            # Filtering the forecast date
            data_fc_iter = data_fc_iter[data_fc_iter[self.date_col] == date_fc]

            # Encoding / scaling features (non-specified columns are dropped, i.e. those that are not used in training)
            data_fc_iter_transf = pd.DataFrame(col_transform.transform(data_fc_iter),
                                               columns=col_transform.get_feature_names_out())
            data_fc_iter_transf = pd.DataFrame(col_scale.transform(data_fc_iter_transf),
                                               columns=col_scale.get_feature_names_out())

            # Storing forecasts in the original dataset
            df_test.loc[df_test[self.date_col] == date_fc, self.target_col] = model.predict(data_fc_iter_transf)
            df_test.loc[df_test[self.date_col] == date_fc, 'day_number'] = idx

            # SHAP values
            if generate_shap:

                # TODO: sklearn XGB doesn't properly for SHAP. For now, only the native `xgb.train()` works.

                tree_explainer = shap.TreeExplainer(model)
                shap_values = tree_explainer.shap_values(data_fc_iter_transf)  # tree_explainer.expected_value
                fc_shap.append(shap_values)
                structure_shap.append(data_fc_iter[[self.date_col, self.id_col]])

        # Keeping relevant columns
        df_test = df_test[[self.date_col, self.id_col, 'day_number', self.target_col]]
        df_test = df_test[df_test[self.date_col] >= self.df_test[self.date_col].min()].reset_index(drop=True)
        df_test['model'] = model_type

        if generate_shap:
            # Renaming columns that appear in both test set and SHAP dataframe
            shap_cols = list(data_fc_iter_transf.columns)
            shap_test_cols = list(data_fc_iter_transf.columns) + list(df_test.columns)
            shap_cols_rename = set(filter(lambda x: shap_test_cols.count(x) > 1, shap_test_cols))
            shap_cols_new = [f'{x}_shap' for x in shap_cols_rename]

            # Concatenating SHAP dataframe with test set
            df_shap = pd.DataFrame(np.concatenate(fc_shap), columns=shap_cols).reset_index(drop=True)
            df_shap = df_shap.rename(columns=dict(zip(shap_cols_rename, shap_cols_new)))
            df_shap = pd.concat([pd.concat(structure_shap).reset_index(drop=True), df_shap], axis=1)
            df_shap = df_test.merge(df_shap, how='left', on=[self.date_col, self.id_col])
        else:
            df_shap = None

        return df_test, df_shap


    def recursive_forecasting_multiple_models(self, model_type_list: list):
        """Looping over multiple models.

        Parameters:
            model_type_list : a list of sklearn ML algorithms.
        """

        # Warning
        print('* When forecasting for multiple models, `generate_shap` == False.')

        fc_models = []

        for model_type in model_type_list:
            print(f'* {model_type}: forecasting...')
            fc, _ = self.recursive_forecasting(model_type=model_type, generate_shap=False)
            fc_models.append(fc)

        fc_models = pd.concat(fc_models)

        return fc_models


if __name__ == '__main__':
    pass