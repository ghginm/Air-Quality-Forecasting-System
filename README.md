# CSCI6409 Project

## Contributors
* [Konstantin Zuev](https://github.com/ghginm) 
* [Mohammed Usama Jasnak](https://github.com/UsamaJasnak/) 
* [Siqi Wang](https://github.com/Ceecee2023)

## Goals

This project targets the critical issue of air pollution by building an Air Quality Forecasting System (AQFS). The objectives include conducting thorough data analysis to understand pollution patterns, creating a precise model for predicting air quality i.e. forecasting PM 2.5 (fine particles in the air that have a diameter of less than 2.5 micrometres), and enhancing the model's performance with advanced feature engineering and hyperparameter tuning.

## Project structure

```
Air-Quality-Forecasting-System/
|
├── data/
├── eda/
│   ├── data_processing_eda.ipynb
│   └── inference_eda.ipynb
├── model/
├── model_creation/
│   ├── feature_engineering.py
│   ├── training_testing.py
│   └── tuning_cv.py
├── utils/
│   └── data_utils.py
├── config.json
├── main.ipynb
├── model_development_inference.py
└── optimisation.py
```

**Additional information**:

* `./data/`: stores all relevant data sources.

* `./eda/data_processing_eda.ipynb`: extensive EDA and data processing.
* `./eda/inference_eda.ipynb`: model performance analysis.

* `./model/`: stores all trained models as well as hyperparameters obainted after tuning.

* `./model_creation/feature_engineering.py`: creating feature engineering to transform hierarchocal time-series problem to ML one.
* `./model_creation/training_testing.py`: training models employing recursive forecasting and obtaining out-of-sample performance on the test set.
* `./model_creation/tuning_cv.py`: tuning models (Optuna, random search).

* `./utils/data_utils.py`: creating data quality reports.

* `./config.json`: a config file with key parameters for tuning, cross-validation and more.
* `./main.ipynb`: interactive notebook with EDA + model training and inference, i.e. a quick and reproducable example.
* `./model_development_inference.py`: model training / testing for a given dataset.
* `./optimisation.py`: model optimisation for a given dataset.
