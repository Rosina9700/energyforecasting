import warnings
import itertools
import pandas as pd
import numpy as np
import csv
import statsmodels.api as sm
from datetime import datetime
from baseline_models import Baseline_average, Baseline_previous, baseline_rolling_predictions, baseline_cross_val_score
from data_wrangling import Results_data, Data_preparation
import sys


def fit_sarimaX(y, arima_params, s_params):
    '''
    Fit a SARIMA model to data with given parameters
    -----------
    y: Pandas Series
    arima_params: Tuple
    s_params: Tuple
    RETURNS:
    -----------
    results: SARIMAResults Class Object
    '''
    print 'SARIMAX: {} x {}'.format(arima_params, s_params)

    mod = sm.tsa.statespace.SARIMAX(y[0], y[1],
                                    order=arima_params,
                                    seasonal_order=s_params,
                                    enforce_stationarity=False,
                                    enforce_invertibility=False,
                                    time_varying_regression=True,
                                    mle_regression=False)

    results = mod.fit()
    return results

def fit_sarima(y, arima_params, s_params):
    '''
    Fit a SARIMA model to data with given parameters
    -----------
    y: Pandas Series
    arima_params: Tuple
    s_params: Tuple
    RETURNS:
    -----------
    results: SARIMAResults Class Object
    '''
    print 'SARIMAX: {} x {}'.format(arima_params, s_params)
    mod = sm.tsa.statespace.SARIMAX(y,
                                    order=arima_params,
                                    seasonal_order=s_params,
                                    enforce_stationarity=False,
                                    enforce_invertibility=False)
    results = mod.fit()
    return results


def rolling_predictions_sarima(y,end,window,params,types=1):
    '''
    Calculating the one-step ahead forecast and rmse for
    a given dataset with both the SARIMA and SARIMAX models.
    -----------
    y: Pandas Series
    end: integer
    window: integer
    params: Tuple
    RETURNS:
    -----------
    forecast: Numpy array
    rmse: float
    model: SARIMAResults Class Object
    '''
    forecast_s = np.zeros(window)
    forecast_sX = np.zeros(window)
    endog = y.ix[:,0]
    exog = sm.add_constant(y.ix[:,1:])
    results = dict()
    for i in xrange(window):
        endog_temp = endog.ix[:end+i]
        exog_temp =  exog.ix[:end+i,:]
        print 'length of cross validation data {}'.format(len(endog_temp))
        model_s, model_sX = None, None
        if types <= 1:
            try:
                model_s = fit_sarima(endog_temp, params[0], params[1])
                forecast_s[i]= model_s.forecast(steps=1).values[0]
            except:
                print 'SKIPPED SARIMA {}-{}'.format(params[0], params[1])


        if types >=1:
            try:
                model_sX = fit_sarimaX((endog_temp, exog_temp), params[0], params[1])
                forecast_sX[i]= model_sX.forecast(steps=1,exog=exog.ix[end+i,:].values.reshape(1,exog.shape[1])).values[0]
            except:
                print 'SKIPPED SARIMAX {}-{}'.format(params[0], params[1])


        # print exog.ix[end+i,:].values.reshape(1,exog.shape[1])
    true = endog[end:end+window].values
    rmse_s = np.sqrt(((true-forecast_s)**2).mean())
    rmse_sX = np.sqrt(((true-forecast_sX)**2).mean())
    results['sarima'] = (forecast_s, rmse_s, model_s)
    results['sarimaX'] = (forecast_sX, rmse_sX, model_sX)
    return results

if __name__ == '__main__':
    project_name, f, season, location = sys.argv[1],sys.argv[2],int(sys.argv[3]),sys.argv[4]
    if sys.argv[5] == 'True':
        T_dependant = True
    else:
        T_dependant = False

    if location == 'local':
        p = '../../capstone_data/Azimuth/clean/{}'.format(project_name)
    else:
        p = project_name

    print'get data for {}....'.format(p)
    dp = Data_preparation(p,f,T_dependant).get_data()
    # df = dp.get_data()
    y = dp.create_variable(agg='sum',feature='power_all')
    # tuned_results = Results_data(project_name)
    # params_s, params_sX = tuned_results.get_data().get_params()
    y_train = y[:-2*season]
    y_test = y[-2*season:]
    cv_folds = 25
    print '\nbaseline - previous...'
    b_previous = Baseline_previous()
    b1_train_rmse, model = baseline_cross_val_score(b_previous, pd.DataFrame(y_train.ix[:,0]), cv_folds, window=1, season=season)
    forecast, b1_test_rmse, model = baseline_rolling_predictions(b_previous, pd.DataFrame(y.ix[:,0]),len(y_train),2*season)
    print 'Baseline-previous train RMSE {}'.format(b1_train_rmse)
    print 'Baseline-previous test RMSE {}'.format(b1_test_rmse)

    b2_train_rmse = None
    b2_test_rmse = None
    print 'baseline - averages....'
    b_average = Baseline_average(freq=f)
    b2_train_rmse, model = baseline_cross_val_score(b_average, pd.DataFrame(y_train.ix[:,0]), cv_folds, window=1, season=season)
    forecast, b2_test_rmse, model = baseline_rolling_predictions(b_average, pd.DataFrame(y.ix[:,0]),len(y_train),2*season)
    print 'Baseline-averages train RMSE {}'.format(b2_train_rmse)
    print 'Baseline-averages test RMSE {}'.format(b2_test_rmse)


    # print '\nFitting Sarima-X models...'
    #
    # # For Sarima model
    # results_s = rolling_predictions_sarima(y,len(y_train),2*season,params_s,types=0)
    # model_s = results_s['sarima'][2]
    #
    # # For SarimaX model
    # results_sX = rolling_predictions_sarima(y,len(y_train),2*season,params_sX,types=2)
    # model_sX = results_sX['sarimaX'][2]
    #
    # print 'Sarima cross val score {}'.format(tuned_results.df.iloc[0]['sarimax'])
    # print'Sarima AIC {}'.format(model_s.aic)
    #
    # print 'SarimaX cross val score {}'.format(tuned_results.df.iloc[2]['sarimax'])
    # print'SarimaX AIC {}'.format(model_sX.aic)
