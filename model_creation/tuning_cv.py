import pickle
import warnings
from typing import Iterable, Literal

import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
from sklearn.model_selection import cross_val_score, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import TargetEncoder, OneHotEncoder
from xgboost import XGBRegressor


## Custom cross-validation

def custom_cv(df: pd.DataFrame, date_col: str = 'date', val_size: int = 7, n_splits: int = 4) -> list:
    """Custom cross-validation for time-series data.
    
    Prameters:
        df : pandas DataFrame with daily time series.
        date_col : the date column (day).
        val_size : the number of days each validation fold will include.
        n_splits : the number of splits that will be performed.
    """

    # Warning
    print('* Prior to using this function, make sure that the index for DataFrame is reset. \n')

    val_end = pd.to_datetime(df[date_col].max())
    cv_idx = []

    for i in range(n_splits, 0, -1):
        tr_threshold = val_end - pd.to_timedelta(val_size*i, unit='d')
        val_threshold = tr_threshold + pd.to_timedelta(val_size, unit='d')

        tr_idx = np.array(df.index[df[date_col] <= tr_threshold])
        te_idx = np.array(df.index[(df[date_col] > tr_threshold) & (df[date_col] <= val_threshold)])

        cv_idx.append((tr_idx, te_idx))

        # Printing additional information
        print(f'* Training (up to): {tr_threshold} | Validation (up to): {val_threshold}')


    return cv_idx

## Splitting data

def train_val_test_split(df: pd.DataFrame, date_col: str = 'date', id_col: str = 'id',
                         test_size: int | None = 8*7, val_size_es: int | None = 2*7,
                         supply_test_dates: bool = False, test_dates: list | None = None,) -> tuple:
    """Splitting data into training, test and validation sets.

    Parameters:
        df : pandas DataFrame with daily time series.
        date_col : the date column (day).
        id_col : a column that uniquely identifies the hierarchy of time series.
        test_size : the number of days in the test set.
        val_size_es : the number of days in the validation set for early stopping.
        supply_test_dates : instead of taking the last n days as the test set, a set of specific dates can be supplied.
        test_dates : a set of specific test dates used if `supply_test_dates` == True.
    """

    df = df.sort_values(by=[id_col, date_col]).reset_index(drop=True)

    if supply_test_dates:
        print("Parameters `test_size` and `val_size_es` will be ignored if `supply_test_dates` == True.")

        df_train = df[df[date_col] < test_dates.min()]
        df_val = None
        df_test = df[df[date_col].isin(test_dates)]
    else:
        test_date = df[date_col].max() - pd.to_timedelta(test_size + 1, unit='d')

        if val_size_es is not None:
            val_date = test_date - pd.to_timedelta(val_size_es + 1, unit='d')

            df_train = df[df[date_col] < val_date].reset_index(drop=True)
            df_val = df[(df[date_col] < test_date) & (df[date_col] >= val_date)].reset_index(drop=True)
            df_test = df[df[date_col] >= test_date].reset_index(drop=True)
        else:
            df_train = df[df[date_col] < test_date].reset_index(drop=True)
            df_val = None
            df_test = df[df[date_col] >= test_date].reset_index(drop=True)
    
    return df_train, df_val, df_test

## Optuna

# TODO: ideally, `cv_early_stopping()`, `oOptunaObjective`, `optuna_tuning()` could be split into two classes with inheritance

def cv_early_stopping(X_train: pd.DataFrame, y_train: np.array, X_eval: pd.DataFrame, y_eval: np.array,
                      model: BaseEstimator, cv: int | Iterable, early_stopping_rounds: int = 10,
                      target_enc_cols: list | None = None, one_hot_cols: list | None = None,
                      baseline_model: bool = False, parallelism: Literal['model', 'cv', 'both'] = 'both') -> tuple:
    """The objective function used by Optuna to guide the optimisation process.

    Parameters:
        X_train : training set with features.
        y_train : the target variable for the training set.
        X_eval : validation set with features.
        y_eval : the target variable for the validation set.
        model : either XGBRegressor or LGBMRegressor.
        cv : sklearn-like cross-validation, i.e. a list of tuples.
        early_stopping_rounds : the maximum number of trees to grow with no improvement.
        target_enc_cols : variables that should be target-encoded.
        one_hot_cols : variables that should be one-hot-encoded.
        baseline_model : get cross-validation results for a model with default hyperparameters.
        parallelism : choosing which stage of the pipeline to parallelise.
    """

    # Type of parallelism
    if parallelism == 'model':
        n_jobs_model = -1
        n_jobs_cv = 1
    elif parallelism == 'cv':
        n_jobs_model = 1
        n_jobs_cv = -1
    elif parallelism == 'both':
        n_jobs_model = -1
        n_jobs_cv = -1
    else:
        raise ValueError('The `parallelism` parameter should be either "model", "cv" or "both".')

    # Creating ColumnTransformer
    transformers = []

    if target_enc_cols is not None:
        transformers.append(('target_encoding', TargetEncoder(target_type='continuous'), target_enc_cols))
    if one_hot_cols is not None:
        transformers.append(('one_hot_encoding', OneHotEncoder(handle_unknown='ignore'), one_hot_cols))

    col_transform = ColumnTransformer(
        transformers=transformers, remainder='passthrough', n_jobs=-1
    )

    # Applying ColumnTransformer outside the pipeline (transforming the validation set)
    X_train = col_transform.fit_transform(X_train, y_train)
    X_eval = col_transform.transform(X_eval)

    # Fitting a model
    if baseline_model:
        if ('XGB' not in type(model).__name__) and ('LGB' not in type(model).__name__):
            raise ValueError('The `model` parameter should be either sklearn LightGBM or XGBoost.')

        best_n_est = None
        model.set_params(**{'n_jobs': n_jobs_model})
    else:
        # Fitting a model for each Optuna trial and getting `n_estimators` using early stopping
        if 'XGB' in type(model).__name__:
            model.set_params(**{'early_stopping_rounds': early_stopping_rounds, 'n_jobs': n_jobs_model})

            # Fitting a model
            model.fit(X_train, y_train, eval_set=[(X_eval, y_eval)], verbose=False)
            best_n_est = model.best_iteration

            # Updating `n_estimators`
            model.set_params(**{'early_stopping_rounds': None, 'n_estimators': best_n_est})
        elif 'LGB' in type(model).__name__:
            model.set_params(**{'early_stopping_round': early_stopping_rounds, 'n_jobs': n_jobs_model, 'verbose': -1})

            # Fitting a model
            model.fit(X_train, y_train, eval_set=[(X_eval, y_eval)])
            best_n_est = model.booster_.best_iteration

            # Updating `n_estimators`
            model.set_params(**{'early_stopping_round': None, 'n_estimators': best_n_est})
        else:
            raise ValueError('The `model` parameter should be either sklearn LightGBM or XGBoost.')

    # Creating the pipeline
    pipeline = Pipeline(
        steps=[
            # ('col_transform', col_transform),
            ('scaler', StandardScaler()),
            ('model', model)
        ]
    )

    # Getting CV scores
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, n_jobs=n_jobs_cv,
                                scoring='neg_root_mean_squared_error')

    mean_cv_score = -1*cv_scores.mean()
    std_cv_score = cv_scores.std()

    if baseline_model:
        print('Avg score (no tuning):', round(mean_cv_score, 3), 'Std score (no tuning):', round(std_cv_score, 3))

    return mean_cv_score, std_cv_score, best_n_est


class OptunaObjective:
    """Defining the hyperparameter space used in Optuna.

    Parameters:
        X_train : training set with features.
        y_train : the target variable for the training set.
        X_eval : validation set with features.
        y_eval : the target variable for the validation set.
        model_type : a type of the gradient boosting.
        cv : sklearn-like cross-validation, i.e. a list of tuples.
        early_stopping_rounds : the maximum number of trees to grow with no improvement.
        target_enc_cols : variables that should be target-encoded.
        one_hot_cols : variables that should be one-hot-encoded.
        parallelism : choosing which stage of the pipeline to parallelise.
    """
    
    def __init__(self, X_train: pd.DataFrame, y_train: np.array, X_eval: pd.DataFrame, y_eval: np.array, 
                 model_type: Literal['xgb', 'lgb'], cv: int | Iterable, early_stopping_rounds: int = 10,
                 target_enc_cols: list | None = None, one_hot_cols: list | None = None,
                 parallelism: Literal['model', 'cv', 'both'] = 'both'):
        self.X_train = X_train
        self.y_train = y_train
        self.X_eval = X_eval
        self.y_eval = y_eval
        self.model_type = model_type
        self.cv = cv
        self.early_stopping_rounds = early_stopping_rounds
        self.target_enc_cols = target_enc_cols
        self.one_hot_cols = one_hot_cols
        self.parallelism = parallelism

    def __call__(self, trial: optuna.trial.FrozenTrial) -> None:
        if self.model_type == 'xgb':
            params = {'booster': trial.suggest_categorical('boosting_type', ['gbtree']),
                      'tree_method': trial.suggest_categorical('tree_method', ['auto']),
                      # For the reg:squarederror objective, the Hessian sum happens to match the number of samples,
                      # but this equality generally does not hold for other objectives.
                      'min_child_weight': trial.suggest_int('min_child_weight', 20, 520, step=50),  #
                      'max_depth': trial.suggest_int('max_depth', 5, 55, step=5),
                      # 'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 4, log=True),  # lambda_l1
                      'reg_lambda': trial.suggest_categorical('reg_lambda', [0.001, 0.01, 0.1, 1, 3, 5]),  # lambda_l2
                      'subsample': trial.suggest_float('subsample', 0.5, 1.0, step=0.1),
                      'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0, step=0.1)}  # feature_fraction
            
            # Defining fixed parameters
            objective = 'reg:squarederror'
            eval_metric = 'rmse'  # early stopping
            learning_rate = 0.05
            n_estimators = 10000

            params['objective'] = objective
            params['eval_metric'] = eval_metric
            params['learning_rate'] = learning_rate
            params['n_estimators'] = n_estimators

            # Model
            model = XGBRegressor(**params)

            # Trial
            trial_cv_score, trial_std_score, best_n_est = cv_early_stopping(X_train=self.X_train, y_train=self.y_train,
                                                                            X_eval=self.X_eval, y_eval=self.y_eval,
                                                                            model=model, cv=self.cv,
                                                                            early_stopping_rounds=self.early_stopping_rounds,
                                                                            target_enc_cols=self.target_enc_cols,
                                                                            one_hot_cols=self.one_hot_cols, baseline_model=False,
                                                                            parallelism=self.parallelism)

            # Saving fixed parameters
            trial.set_user_attr('objective', objective)
            trial.set_user_attr('eval_metric', eval_metric)
            trial.set_user_attr('learning_rate', learning_rate)
            trial.set_user_attr('n_estimators', best_n_est)

            trial.set_user_attr('value_std', trial_std_score)

        elif self.model_type == 'lgb':
            params = {'boosting_type': trial.suggest_categorical('boosting_type', ['gbdt']),
                      'min_child_samples': trial.suggest_int('min_child_samples', 20, 420, step=50),  # min_data_in_leaf
                      'num_leaves': trial.suggest_int('num_leaves', 50, 650, step=60),
                      'max_depth': trial.suggest_int('max_depth', 5, 20, step=3),
                      # 'reg_alpha': trial.suggest_float('reg_alpha', 0.001, 4, log=True),  # lambda_l1
                      'reg_lambda': trial.suggest_categorical('reg_lambda', [0.001, 0.01, 0.1, 1, 3, 5]),  # lambda_l2
                      'subsample': trial.suggest_float('subsample', 0.5, 1.0, step=0.1),
                      'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0, step=0.1)}  # feature_fraction
            
            # Defining fixed parameters
            feature_pre_filter = False
            boost_from_average = True
            subsample_freq = 1
            objective = 'rmse'
            metric = 'rmse'  # early stopping
            learning_rate = 0.05
            n_estimators = 10000

            params['feature_pre_filter'] = feature_pre_filter
            params['boost_from_average'] = boost_from_average
            params['subsample_freq'] = subsample_freq
            params['objective'] = objective
            params['metric'] = metric
            params['learning_rate'] = learning_rate
            params['n_estimators'] = n_estimators

            # Model
            model = LGBMRegressor(**params)

            # Trial
            trial_cv_score, trial_std_score, best_n_est = cv_early_stopping(X_train=self.X_train, y_train=self.y_train,
                                                                            X_eval=self.X_eval, y_eval=self.y_eval,
                                                                            model=model, cv=self.cv,
                                                                            early_stopping_rounds=self.early_stopping_rounds,
                                                                            target_enc_cols=self.target_enc_cols,
                                                                            one_hot_cols=self.one_hot_cols, baseline_model=False,
                                                                            parallelism=self.parallelism)
            
            # Saving fixed parameters
            trial.set_user_attr('feature_pre_filter', feature_pre_filter)
            trial.set_user_attr('boost_from_average', boost_from_average)
            trial.set_user_attr('subsample_freq', subsample_freq)
            trial.set_user_attr('objective', objective)
            trial.set_user_attr('metric', metric)
            trial.set_user_attr('learning_rate', learning_rate)
            trial.set_user_attr('n_estimators', best_n_est)

            trial.set_user_attr('value_std', trial_std_score)
            
        else:
            raise ValueError('The `model_type` parameter should be either "xgb" or "lgb".')
        
        return trial_cv_score


class callback_optuna_score:
    def __init__(self, n_trials):
        self.n_trials = n_trials

    def __call__(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial) -> None:
        print(trial.number + 1, '/', self.n_trials, 'Avg score (optuna):', round(trial.value, 3),
              'Std score (optuna):', round(trial.user_attrs['value_std'], 3))


def optuna_tuning(X_train: pd.DataFrame, y_train: np.array, X_eval: pd.DataFrame, y_eval: np.array, path: str,
                  model_type: Literal['xgb', 'lgb'], cv: int | Iterable, early_stopping_rounds: int = 10,
                  target_enc_cols: list | None = None, one_hot_cols: list | None = None,
                  parallelism: Literal['model', 'cv', 'both'] = 'both', optuna_n_trials: int = 50):
    """Tuning models with Optuna.

    Parameters:
        X_train : training set with features.
        y_train : the target variable for the training set.
        X_eval : validation set with features.
        y_eval : the target variable for the validation set.
        path : a directory where to store final parameters.
        model_type : a type of the gradient boosting.
        cv : sklearn-like cross-validation, i.e. a list of tuples.
        early_stopping_rounds : the maximum number of trees to grow with no improvement.
        target_enc_cols : variables that should be target-encoded.
        one_hot_cols : variables that should be one-hot-encoded.
        parallelism : choosing which stage of the pipeline to parallelise.
        optuna_n_trials : the number of tuning iterations.
    """

    # Warning
    if parallelism == 'both':
        warnings.warn('Using `parallelism` == "both" can lead to significant slowdowns in some cases.')

    # Verbosity
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # Initiating Optuna study
    study = optuna.create_study(direction='minimize')

    study.optimize(OptunaObjective(
            X_train=X_train, y_train=y_train, X_eval=X_eval, y_eval=y_eval,
            model_type=model_type, cv=cv, early_stopping_rounds=early_stopping_rounds,
            target_enc_cols=target_enc_cols, one_hot_cols=one_hot_cols, parallelism=parallelism),
        callbacks=[callback_optuna_score(n_trials=optuna_n_trials)], n_trials=optuna_n_trials
    )

    # Retrieving optimisation results
    all_params = study.trials_dataframe()

    # Saving optuna results
    all_params.to_excel(f'{path}\\optuna_{model_type}.xlsx', index=False)

    # Retrieving fixed parameters
    fixed_params = all_params.loc[all_params['value'].idxmin(), [x for x in all_params.columns if 'user_attrs' in x]]
    fixed_params.index = [x.replace('user_attrs_', '') for x in fixed_params.index]

    # Mean and std scores
    mean_std_scores = all_params.loc[all_params['value'].idxmin(), ['value', 'user_attrs_value_std']]

    # Retrieving the best parameters
    params_final = study.best_params
    params_final.update(fixed_params)
    del params_final['value_std']

    # Saving final parameters
    with open(f'{path}\\params_final_{model_type}.pickle', 'wb') as handle:
        pickle.dump(params_final, handle)

    return mean_std_scores['value'], mean_std_scores['user_attrs_value_std']

## Random search

def random_search_tuning(X_train: pd.DataFrame, y_train: np.array, path: str,
                         model_type: Literal['lasso', 'rf'], cv: int | Iterable,
                         target_enc_cols: list | None = None, one_hot_cols: list | None = None,
                         parallelism: Literal['model', 'cv', 'both'] = 'both', n_iter: int = 50) -> tuple:
    """Tuning models with random search.

    Parameters:
        X_train : training set with features.
        y_train : the target variable for the training set.
        path : a directory where to store final parameters.
        model_type : a type of sklearn ML algorithm.
        cv : sklearn-like cross-validation, i.e. a list of tuples.
        target_enc_cols : variables that should be target-encoded.
        one_hot_cols : variables that should be one-hot-encoded.
        parallelism : choosing which stage of the pipeline to parallelise.
        n_iter : the number of tuning iterations.
    """

    # Type of parallelism
    if parallelism == 'model':
        n_jobs_model = -1
        n_jobs_cv = 1
    elif parallelism == 'cv':
        n_jobs_model = 1
        n_jobs_cv = -1
    elif parallelism == 'both':
        n_jobs_model = -1
        n_jobs_cv = -1
        warnings.warn('Using `parallelism` == "both" can lead to significant slowdowns in some cases.')
    else:
        raise ValueError('The `parallelism` parameter should be either "model", "cv" or "both".')

    # Creating ColumnTransformer
    transformers = []

    if target_enc_cols is not None:
        transformers.append(('target_encoding', TargetEncoder(target_type='continuous'), target_enc_cols))
    if one_hot_cols is not None:
        transformers.append(('one_hot_encoding', OneHotEncoder(handle_unknown='ignore'), one_hot_cols))

    col_transform = ColumnTransformer(
        transformers=transformers, remainder='passthrough', n_jobs=-1
    )

    # Initialising a model
    if model_type == 'lasso':
        param_distributions = {
            'model__alpha': [x / 1000 for x in range(1, 100, 2)] + [x / 10 for x in range(1, 100, 2)]
        }

        model = Lasso(max_iter=2000)

        n_jobs_cv = -1
        warnings.warn('For Lasso regression `parallelism` will always be set to "cv".')
    elif model_type == 'rf':
        param_distributions = {
            'model__n_estimators': [100, 150, 200, 300],
            'model__max_depth': [5, 10, 15, 20, 30, 50],
            'model__min_samples_split': [50, 100, 150, 200, 300, 500],  # splitting nodes
            'model__min_samples_leaf': [25, 50, 75, 100, 150, 250]  # observations per leaf
        }

        model = RandomForestRegressor()
        model.set_params(**{'n_jobs': n_jobs_model})
    else:
        raise ValueError('The `model_type` parameter should be either "lasso" or "rf".')

    # Creating the pipeline
    pipeline = Pipeline(
        steps=[
            ('col_transform', col_transform),
            ('scaler', StandardScaler()),
            ('model', model)
        ]
    )

    # Random search
    random_search = RandomizedSearchCV(pipeline, param_distributions=param_distributions, n_iter=n_iter, cv=cv,
                                       n_jobs=n_jobs_cv, scoring='neg_root_mean_squared_error', verbose=1, refit=False)

    random_search.fit(X_train, y_train)

    # Mean and std scores
    mean_score = round(-1*random_search.best_score_, 3)
    std_score = round(random_search.cv_results_['std_test_score'][random_search.best_index_], 3)
    print('Avg score (random search):', mean_score, 'Std score (random search):', std_score)

    # Saving final parameters
    model_params = random_search.best_params_
    model_params = {key.replace('model__', ''): value for key, value in model_params.items()}

    with open(f'{path}\\params_final_{model_type}.pickle', 'wb') as handle:
        pickle.dump(model_params, handle)

    return mean_score, std_score


if __name__ == '__main__':
    pass