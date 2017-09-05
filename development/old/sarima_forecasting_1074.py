import warnings
import itertools
import pandas as pd
import numpy as np
import statsmodels.api as sm
from datetime import datetime
from baseline_models import Baseline_average, Baseline_previous

def get_data(project_name):
    '''
    Read in the featurized data for the given project_name
    PARAMETERS:
    -----------
    project_name: String
    RETURNS:
    -----------
    df: Pandas DataFrame with DatetimeIndex
    '''
    filelocation='{}_featurized.csv'.format(project_name)
    # filelocation='{}_featurized.csv'.format(project_name)
    df = pd.read_csv(filelocation)
    df['t'] = pd.to_datetime(df['t'], format='%Y-%m-%d %H:%M:%S')
    df.set_index('t',inplace=True)
    df = calculate_power(df)
    return df

def calculate_power(df):
    '''
    Calculate total power for that site
    PARAMETERS:
    -----------
    df: Pandas DataFrame with DatetimeIndex
    RETURNS:
    -----------
    df: Pandas DataFrame with DatetimeIndex
    '''
    df['power_1'] = df['load_v1rms'] * df['load_i1rms']
    df['power_2'] = df['load_v2rms'] * df['load_i2rms']
    df['power_3'] = df['load_v3rms'] * df['laod_i3rms']
    df['power_all'] = df['power_1'] +df['power_2']+df['power_3'] * 5./12
    return df

def get_ready_for_sarima(df, feature, freq='H'):
    '''
    Calculate total power for that site
    PARAMETERS:
    -----------
    df: Pandas DataFrame with DatetimeIndex
    feature: String
    freq: String following panda resample frequency nomenclature
    RETURNS:
    -----------
    y: Pandas DataFrame with DatetimeIndex
    '''
    y = df[feature]
    y = y.fillna(y.bfill())
    y = y.resample(freq).sum()
    return pd.DataFrame(y)
    
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
        model = model.fit(y)
        forecast[i]= model.forecast(steps=1)[0]
    true = y[end:end+window].values
    rmse = np.sqrt(((true-forecast)**2).mean())
    return forecast, rmse, model

def baseline_cross_val_score(model, y, chunks, window=1):
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
    length = len(y)-window
    chunk_size = length/chunks
    rmses = []
    for i in xrange(chunks):
        end_index = (i+1)*chunk_size
        forecast, rmse, model = baseline_rolling_predictions(model, y,end_index,window)
        rmses.append(rmse)
    return np.asarray(rmses).mean(), model


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

def rolling_predictions_sarima(y,end,window,params):
    '''
    Calculating the one-step ahead forecast and rmse for
    a given dataset and SARIMA model.
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
        y_temp = y[0:end+i]
        try:
            model = fit_sarima(y_temp,params[0],params[1])
        except:
            print 'SKIPPED {}-{}'.format(params[0], params[1])
            continue
        forecast[i]= model.forecast(steps=1).values[0]
    true = y[end:end+window].values
    rmse = np.sqrt(((true-forecast)**2).mean())
    return forecast, rmse, model


def cross_val_score(y, params, chunks, window=1):
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
    length = len(y)-window
    chunk_size = length/chunks
    rmses = []
    for i in xrange(chunks):
        end_index = (i+1)*chunk_size
        forecast, rmse, model = rolling_predictions_sarima(y,end_index,window, params)
        rmses.append(rmse)
    return np.asarray(rmses), model

def cross_validation_sarima(y, param, param_seasonal, k):
    '''
    Calls the cross_val_score function to conduction cross validation and return
    the average rmse for the given model
    -----------
    y: Pandas Series
    param: Tuple
    param_seasonal: Tuple
    k: integer
    RETURNS:
    -----------
    reults: Tuple(SARIMAXResults Object, float)
    '''
    rmses, model = cross_val_score(y, (param, param_seasonal), chunks=k)
    ave_rmse = rmses.mean()
    return (model, ave_rmse)

def grid_search_sarima(y, pdq, seasonal_pdq, k):
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
    results = []
    for param in pdq:
        for param_seasonal in seasonal_pdq:
            temp_results = cross_validation_sarima(y, param, param_seasonal, k)
            results.append(temp_results)
    return results

def find_best_sarima(y, params, season, k=10):
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
    s_pdq = list(itertools.product(range(0,2), range(0,2)))
    seasonal_pdq = [(x[0], 0, x[1], season) for x in s_pdq]
    warnings.filterwarnings("ignore") # specify to ignore warning messages
    results = grid_search_sarima(y, pdq, seasonal_pdq, k)
    top_ind = np.array([r[1] for r in results]).argmin()
    return results[top_ind][0], results[top_ind][1]


if __name__== '__main__':
    projects = ['project_1074']
    for p in projects:
        print'get data for {}....'.format(p)
        project_name = p
        df = get_data(project_name)
        y = get_ready_for_sarima(df,freq='H', feature='power_all')
        y_train = y[:-24]
        y_test = y[-24:]
        cv_folds = 6
        #
        print '\nbaseline - previous...'
        b_previous = Baseline_previous()
        b1_train_rmse, model = baseline_cross_val_score(b_previous, y_train, cv_folds)
        forecast, b1_test_rmse, model = baseline_rolling_predictions(b_previous, y,len(y_train)-24,24)
        print 'Baseline-previous train RMSE {}'.format(b1_train_rmse)
        print 'Baseline-previous test RMSE {}'.format(b1_test_rmse)
        #
        print 'baseline - averages....'
        b_average = Baseline_average()
        b2_train_rmse, model = baseline_cross_val_score(b_average, y_train, cv_folds)
        forecast, b2_test_rmse, model = baseline_rolling_predictions(b_average, y,len(y_train)-24,24)
        print 'Baseline-averages train RMSE {}'.format(b2_train_rmse)
        print 'Baseline-averages test RMSE {}'.format(b2_test_rmse)

        print '\nfind best sarima...'
        y_train = y[:-24]
        y_test = y[-24:]
        p = range(0,4)
        q = range(1,3)
        d = range(0,2)
        # p = range(1,2)
        # q = range(0,1)
        # d = range(0,1)
        params = (p,d,q)
        model, s_train_rmse = find_best_sarima(y_train,params,24, k=cv_folds)
        best_params = model.specification
        params = (best_params['order'],best_params['seasonal_order'])
        test_forecast, s_test_rmse, model = rolling_predictions_sarima(y,len(y_train)-24,24,params)
        print('SARIMA{}x{}{} - AIC:{}'.format(best_params['order'], best_params['seasonal_order'],
                                         best_params['seasonal_periods'],model.aic))
        print 'Training cross validation RMSE: {}'.format(s_train_rmse)
        print'Test cross validation RMSE {}'.format(s_test_rmse)
        # #
        now = datetime.now().strftime('%m_%d_%H_%M_%S')
        filename = 'output_{}_{}.txt'.format(project_name,now)
        test = 1
        train = 0
        with open(filename, "w") as text_file:
            train_results = '{},{},{},{},{},{}'.format(project_name,train,b1_train_rmse,b2_train_rmse,s_train_rmse,best_params)
            test_results = '{},{},{},{},{},{}'.format(project_name,test,b1_test_rmse,b2_test_rmse,s_test_rmse,best_params)
            string_to_write = 'project,test,baseline_previous,baseline_averages,sarima,sarima_params\n{}\n{}'.format(train_results,test_results)
            text_file.write(string_to_write)
