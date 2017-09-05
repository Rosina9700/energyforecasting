import pandas as pd
import numpy as np
import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

class Baseline_average(object):
    def __init__(self, freq):
        self.averages = None
        self.freq = freq
        self.score_ = None
        self.y_train = None

    def fit(self, y):
        self.y_train = y
        dateindex = y.index
        y['dayofweek'] = dateindex.dayofweek
        if self.freq == 'H':
            y['hour'] = dateindex.hour
            self.averages = y.groupby(['dayofweek','hour'])['power_all'].mean()
            y.drop(['dayofweek','hour'],axis=1, inplace=True)
        elif self.freq == 'D':
            self.averages = y.groupby('dayofweek')['power_all'].mean()
            y.drop('dayofweek',axis=1, inplace=True)
        return self

    def predict(self, start, periods):
        date_index = pd.date_range(start,periods=periods,freq=self.freq)
        if self.freq == 'H':
            predictions = pd.Series(date_index).apply(lambda x: self.averages[x.dayofweek][x.hour])
        elif self.freq =='D':
            predictions = pd.Series(date_index).apply(lambda x: self.averages[x.dayofweek])
        return predictions

    def score(self, y):
        predictions = self.predict(y.index.min().strftime('%Y-%m-%d %H:%M:00'), len(y)).values
        true = y.values
        rmse = np.sqrt(((true - predictions)**2).mean())
        return rmse

    def forecast(self,steps):
        start_date = self.y_train.index.max()
        forecasts = np.zeros(steps)
        for i in xrange(1,steps+1):
            if self.freq =='H':
                next_step = start_date + pd.Timedelta(hours=i)
                dayofweek = next_step.dayofweek
                hour = next_step.hour
                pred = self.averages[dayofweek][hour]
            elif self.freq =='D':
                next_step = start_date + pd.Timedelta(days=i)
                dayofweek = next_step.dayofweek
                pred = self.averages[dayofweek]
            forecasts[i-1] = pred
        return forecasts


class Baseline_previous(object):
    def __init__(self):
        self.y_train = None

    def fit(self, y):
        self.y_train = y
        return self

    def forecast(self, steps):
        forecasts = np.zeros(steps)
        for i in xrange(0,steps):
            forecasts[i] = self.y_train.values[-1]
        return forecasts

def baseline_rolling_predictions(model, y, end, window):
    '''
    Calculate rolling forecasts and their rmse for the baseline class
    defined in baseline_models.py
    -----------
    model: Baseline Class Object
    y: Pandas Series
    end: Integer
    RETURNS:
    -----------
    forecast: Numpy array
    rmse: float
    model: Baseline Class Object
    '''
    forecast = np.zeros(window)
    for i in xrange(window):
        y_temp = y[0:end+i]
        model = model.fit(y_temp)
        forecast[i]= model.forecast(steps=1)[0]
    true = y[end:end+window].values
    forecast = forecast.reshape(window,1)
    rmse = np.sqrt(((true-forecast)**2).mean())
    return forecast, rmse, model

def baseline_cross_val_score(model, y, chunks, window, season):
    '''
    Calculates the cross validation score for Baseline models according to the
    format used for evaluating SARIMA models.
    -----------
    model: Baseline Class Object
    y: Pandas Series
    chunks: integer
    window: integer
    RETURNS:
    -----------
    rmse: float
    model: Baseline Class Object
    '''
    length = len(y.ix[:,0])-window
    start = max(length/2, season)
    chunks = min(length-start, chunks)
    chunk_size = (length-start)/chunks
    rmses = []
    print length, start, chunks, chunk_size
    for i in xrange(chunks+1):
        end_index = start + (i)*chunk_size
        forecast, rmse, model = baseline_rolling_predictions(model, y,end_index,window)
        rmses.append(rmse)
    return np.asarray(rmses).mean(), model

def baseline_forecasts(y, window, f):
    '''
    Gives the rolling predictions over a given for both baseline model
    -----------
    model: Baseline Class Object
    y: Pandas Series
    window: integer
    RETURNS:
    -----------
    forecasts_b1: Numpy array
    forecasts_b2: Numpy array
    '''
    print '\nbaseline - previous...'
    b_previous = Baseline_previous()
    forecast_b1, b1_test_rmse, model = baseline_rolling_predictions(b_previous, pd.DataFrame(y.ix[:,0]),len(y)-window,window)
    print 'Baseline-previous test RMSE {}'.format(b1_test_rmse)

    print 'baseline - averages....'
    b_average = Baseline_average(f)
    forecast_b2, b2_test_rmse, model = baseline_rolling_predictions(b_average, pd.DataFrame(y.ix[:,0]),len(y)-window,window)
    print 'Baseline-averages test RMSE {}'.format(b2_test_rmse)

    return forecast_b1, forecast_b2
