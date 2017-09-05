# Energy Demand Forecasting using Sarimax models

## Introduction
Distributed Energy Resources such as solar and storage are disrupting our traditional
energy supply and to ensure that they are being used to their full potential,
energy demand forecasting is critical. Knowledge of future energy demand allows
'smart' energy system to make the correct dispatch strategy decision in the present moment,
offering the most value to the customers.

## Data
I am working with 5 minute energy demand data (voltage and current for all three phases) from
an energy service provider currently operating in Kenya and Ghana. This project also uses
NASA meteorological data from the MERRA2 dataset.

## Approach
In the study, I developed a pipeline that:
- cleans load data (involves manual examination of the data)
- creates and adds relevant features
- evaluates the performance of a range of forecasting models on one-step-ahead forecasting
  - Baseline 1: Forecast previous energy demand
  - Baseline 2: Forecast average energy demand for that period in Time
  - Sarima : Forecast based of a tuned Sarima model
  - Sarimax : Forecast based on a tuned SarimaX with fixed beta coefficients for the linear regression
  - Sarimax_variable : Forecast based on a tuned SarimaX with variable beta coefficients for the linear regression

## How to use this repo:
**(1) Data Collection**

Energy demand should be in csv format with a column which includes the timestamp
for each data point.

NASA MERRA2 data can be collected using the download.ipynb in the *merra_data* folder. Note that this data is available with one week latency and therefore if the forecasting code is to be used in production, there must be real-time measure climate data. MERRA data is used in this project to demonstrate the impact of weather related features on energy demand forecasting.
The download notebook has been developed by the [Open Power System Data](https://open-power-system-data.org/) program with additional
meteorological features added for this project (such as cloud cover, humidity and irradiance).
The repo for this project can be found [here](https://github.com/Open-Power-System-Data/weather_data/blob/2017-07-05/main.ipynb)

**(2) Data Cleaning**

In the *data_cleaning* directory, you will find a template cleaning jupyter notebook.
Load your energy demand data and inspect the continuity and data quality. Remove any sections of bad data keeping in mind that timeseries models work best with continuous. Small gaps can be interpolated however it is best to avoid
large gaps in the data.

**(3) Create features**

In the *scripts* directory, you will find a script named 'feature_engineering.py'. Change the path in the script so that it points to the directory containing the cleaned energy demand data created in step (2). Call this script in the terminal as follows:

**python feature_engineering.py [project_name] [y or n]**
- The y or n refers to whether or not the data contains outage events that need to be 'smoothed' (interpolated using none outage events).

This script will create various lagged features and add the weather data to the energy demand data. There are a range of features created which are not used directly in this analysis. This is because this script is also used to create variables for investigating power outages.

**(4) Run the model evaluation scipt**

In the *scripts* directory, you will find a script named 'main.py'. Change the path in the script so that it points to the directory containing the featurized energy demand and weather data created in step(3). Also ensure that the name of the feature to be forecasted is correct in the creation of the dependant variable. Call this script in the terminal as follows:

**python main.py [project_name] [frequency] [season]**

 where project_name = the name included in the data filename, frequency is either 'D' for daily or 'H' for hourly and season is an integer representing the seasonal period for the sarimax moodel.

The scipts will then run a time series cross validation (across 25 folds) for the 5 models above. It will create a csv file in the current directory with the results of the analysis. It will include the cross validation RMSE for one-step ahead forecasting for all models as well as the parameters for the best models. This will allow you to compare different models and choose which one is best for you moving forwards.

**(5) Looking closer at final models**

The daily_results and hourly_results notebooks found in the *scripts* folder provide an easy template to load in the results of your analysis and evaluate the performance of the forecasting models in more detail.
