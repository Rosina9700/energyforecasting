import warnings
import itertools
import pandas as pd
import numpy as np
import csv
import statsmodels.api as sm
from collections import defaultdict
from datetime import datetime
from baseline_models import Baseline_average, Baseline_previous, baseline_rolling_predictions, baseline_cross_val_score
from data_wrangling import Results_data, Data_preparation
import sys


class Sarima_predictions(object):
    def __init__(self, params, model_type):
        self.params = (params[0],params[1])
        self.model_type = model_type
        model_names = ['sarima','sarimax','sarimax_v']
        self.m_name = model_names[self.model_type]
        self.forecast = None

    def fit(self, y):
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
        endog = y.iloc[:,0]
        exog = sm.add_constant(y.ix[:,1:])
        if self.model_type == 0:
            mod = sm.tsa.statespace.SARIMAX(endog,
                                            order=self.params[0],
                                            seasonal_order=self.params[1],
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)

        elif self.model_type == 1:
            mod = sm.tsa.statespace.SARIMAX(endog, exog,
                                            order=self.params[0],
                                            seasonal_order=self.params[1],
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)
        elif self.model_type == 2:
            mod = sm.tsa.statespace.SARIMAX(endog, exog,
                                            order=self.params[0],
                                            seasonal_order=self.params[1],
                                            enforce_stationarity=False,
                                            enforce_invertibility=False,
                                            time_varying_regression=True,
                                            mle_regression=False)


        results = mod.fit()
        return results

    def rolling_predictions_sarima(self, y,end,window):
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
        forecast = np.zeros(window)
        for i in xrange(window):
            y_temp = y.iloc[:end+i]
            model = None
            try:
                print self.params
                print self.model_type
                model = self.fit(y_temp)
                forecast_s[i]= model_s.forecast(steps=1).values[0]
            except:
                print 'SKIPPED SARIMA {}-{}'.format(self.params[0], self.params[1])
            # print exog.ix[end+i,:].values.reshape(1,exog.shape[1])
        true = y.iloc[end:end+window,0].values
        rmse = np.sqrt(((true-forecast)**2).mean())
        results = (forecast, rmse, model)
        self.forecast = forecast
        return results

    def cross_val_score(self,y, chunks, window=1):
        '''
        Break a training set into chunks and calcualtes the average
        rmse from forecasts. The training set gradually grow by size chunk at
        each iteration.
        -----------
        y: Pandas Series
        params: Tuple
        chunks: integer
        window: integer
        RETURNS:
        -----------
        rmse: float
        model: SARIMAResults Class Object
        '''
        season = self.params[1][-1]
        length = len(y.ix[:,0])-window
        start = max(length/2, season)
        chunks = min(length-start, chunks)
        chunk_size = (length-start)/chunks
        rmses = []
        for i in xrange(chunks+1):
            end_index = (start) + (i)*chunk_size
            print 'data length: {} chunks size: {}'.format(length, end_index)
            results = self.rolling_predictions_sarima(y,end_index,window)
            rmses.append(results[1])
        return (np.asarray(rmses).mean(), results[2])

def grid_search_sarima(y, pdq, seasonal_pdq, combination, k):
    '''
    For the pdq's and seasonal_pdq's provided, fit every possible model
    and cross validate with a k chunks.
    -----------
    y: Pandas Series
    param: List of Tuples
    param_seasonal: List of Tuples
    k: Integer
    RETURNS:
    -----------
    results: List of Tuples
    '''
    print 'number of models {}'.format(len(pdq)*len(seasonal_pdq))
    results = defaultdict(list)
    for param in pdq:
        for param_seasonal in seasonal_pdq:
            for combo in combination:
                sp = Sarima_predictions((param, param_seasonal),combo)
                m_name = sp.m_name
                res = sp.cross_val_score(y, k)
                results[m_name].append(res)
    return results

def find_best_sarima(y, params, season, combination, k=10):
    '''
    Grid search over every possible combination of p,d,q and season provided. In
    the cross validation, use k chunks to calculate rmse.
    -----------
    y: Pandas Series
    param: Tuple
    season: Inter
    k: Integer
    RETURNS:
    -----------
    results: SARIMAXResults Object, float
    '''
    pdq = list(itertools.product(params[0], params[1], params[2]))
    s_pdq = list(itertools.product(range(0,2), range(0,2), range(0,2)))
    if season == 7:
        seasonal_pdq = [(x[0], x[1], x[2], season) for x in s_pdq]
    else:
        seasonal_pdq = [(x[0], 0, x[2], season) for x in s_pdq]
    warnings.filterwarnings("ignore") # specify to ignore warning messages
    results = grid_search_sarima(y, pdq, seasonal_pdq, combination, k)
    # results_s, results_sX = grid_search_sarima(y, [(0,1,1)], [(1,0,1,season)], k)
    best = defaultdict(list)
    for key, values in results.iteritems():
        top_ind = np.nanargmin(np.array([r[0] for r in values]))
        best_res = values[top_ind]
        best[key].append(best_res[1])
        best[key].append(best_res[0])

    return best
