import numpy as np
import pandas as pd
from datetime import datetime
import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

def get_clean_data(path,time_col):
    '''
    Read in the clean data from the specified path. Pass through the column name
    which contains the timestamp
    PARAMETERS
    ----------
    path: String
    time_col: String
    RETURNS
    --------
    df: Pandas DataFrame with DatetimeIndex
    '''
    df = pd.read_csv(path)
    df['t'] = pd.to_datetime(df[time_col], format='%Y-%m-%d %H:%M:%S')
    df.set_index('t',inplace=True)
    return df

def get_relay_start(df):
    '''
    Create a flag to say whether an outage started that hour.
    PARAMETERS
    ----------
    df: Pandas Dataframe
    RETURNS
    --------
    df: Pandas DataFrame with DatetimeIndex
    '''
    df['begin_gen'] = df.relay_est.shift(1)
    df['begin_gen'] = df['relay_est']-df['begin_gen']
    df['begin_gen'] = df['begin_gen'].apply(lambda x: 1 if x== 1 else 0)
    return df

def check_time_shift(timestamps, delta, expected_shift=5):
    '''
    Creates a column which flags if the time shift creates time delta which are
    not as expected. Flags data points where continuity is broken
    PARAMETERS
    ----------
    timestamps: Pandas DatetimeIndex
    delta: unit time shift to be applied
    expected_shift: Expected time shift in minutes
    RETURNS
    --------
    time_diff_col_name: Column name specific to this time shift
    time_diff.values: Pandas Series of flag values
    '''
    time_diff = (pd.Series(timestamps) - pd.Series(timestamps).shift(delta))
    time_string = '0 days 00:0{}:00'.format(delta*expected_shift)
    time_diff = time_diff.apply(lambda x: None if x!= pd.to_timedelta(time_string) else 1)
    time_diff_col_name = 't_diff-{}'.format(delta)
    return time_diff_col_name, time_diff.values

def shift_features(df, features, deltas):
    '''
    Create time shifted features for each data point.
    PARAMETERS
    ----------
    df: Pandas DataFrame with columns to shift
    features: List of column names to shift
    deltas: List of time shifts to implement
    RETURNS
    --------
    new_df: Pandas DataFrame with all shifted features added
    '''
    shifted_dfs = [df]
    for d in deltas:
        # Shift features and create new column names
        shifted = df[features].shift(d)
        col_names = []
        for f in features:
            name = '{}-{}'.format(f,d)
            col_names.append(name)
        shifted.columns = col_names
        # Check for time shift continuity
        col_name, values = check_time_shift(df.index, d)
        shifted[col_name] = values
        # Append shifted df to list of dfs
        shifted_dfs.append(shifted)
    # Concatenate all new shifted dfs to original df
    print 'num of shifted columns {}'.format(len(shifted_dfs))
    new_df = pd.concat(shifted_dfs,axis=1)
    # Drop columns with Na's (where continuity has been broken)
    new_df.dropna(inplace=True)
    return new_df

def create_dummies(df, columns):
    for c in columns:
        df = pd.get_dummies(df, columns=[c])
    return df


def get_weather_data():
    '''
    Read in the merra weather data for 2016 and 2017, saved as csv
    RETURNS
    --------
    weather: Pandas DataFrame with all weather data
    '''
    weather_16 = pd.read_csv('merra_data/accra_2016/weather_data_Accra_2016.csv')
    weather_17 = pd.read_csv('merra_data/accra_2017/weather_data_Accra_2017.csv')
    # append 2017 to 2016
    weather = weather_16.append(weather_17, ignore_index=True)
    # create column with datetime timestamp
    weather['timestamp'] = pd.to_datetime(weather['timestamp'], format='%Y-%m-%d %H:%M:%S')
    weather.sort_values('timestamp',inplace=True)
    weather = weather.set_index('timestamp')
    weather['year'] = weather.index.year
    weather['month'] = weather.index.month
    weather['day'] = weather.index.day
    weather['hour'] = weather.index.hour
    weather = pd.DataFrame(weather.groupby(['year','month','day','hour']).mean())
    weather['expanded_date'] = weather.index.map(lambda x: datetime(x[0], x[1], x[2],x[3]))
    weather = weather.set_index('expanded_date')
    return weather

def add_weather_data(left, right):
    '''
    Left join between the energy demand data and the weather data. Handles the
    scenario where the frequency of energy data is less than the weather
    PARAMETERS
    ----------
    left: Pandas DataFrame
    right: Pandas DataFrame
    RETURNS
    --------
    weather: Pandas DataFrame with all weather data
    '''
    left['datetime_hr'] = left.index.values
    left['datetime_hr'] = left['datetime_hr'].apply(lambda x: datetime(x.year, x.month, x.day, x.hour))
    new_df = left.join(right,how='left',on=['datetime_hr'])
    new_df.dropna(inplace=True)
    return new_df

def time_window_aggregate(df, feature,func,time_window):
    '''
    Calculates the aggregate value of a column for a given time_window starting
    before each point
    PARAMETERS
    ----------
    df: Pandas DataFrame
    feature: column name of feature to shift
    time_window: time window to aggregate over, use convention used in Pandas
                 reindex or rolling:http://pandas.pydata.org/pandas-docs/stable/timeseries.html#offset-aliases
    func: Aggregate function to use
    '''
    temp = pd.DataFrame(df[feature].shift(1))
    if func == 'sum':
        result = temp.rolling(time_window,min_periods=3).sum()
    elif func == 'std':
        result = temp.rolling(time_window,min_periods=3).std()
    elif func == 'mean':
        result = temp.rolling(time_window,min_periods=3).mean()
    elif func == 'min':
        result = temp.rolling(time_window,min_periods=3).min()
    elif func == 'max':
        result = temp.rolling(time_window,min_periods=3).max()
    else:
        print 'Aggregate function {} does not exist'.format(func)
        return None
    col_name = '{}_{}_{}'.format(feature,time_window,func)
    df[col_name] = result
    return df

def create_time_aggregates(df,params):
    '''
    For the dataframe provided and parameter list with instructions, create new
    features of time aggregate values
    PARAMETERS
    ----------
    df: Pandas DataFrame
    Params: Nested dictionary
    RETURNS
    --------
    df: Pandas DataFrame
    '''
    for key, values in params.iteritems():
        for k, v in values.iteritems():
            for f in v:
                df = time_window_aggregate(df,f,k,key)
    return df

def outage_smoothing(df, feature, time_window):
    '''
    For sites where there are outages and no generator, we may want to smooth over
    the outage periods so that they are low power events or NaNs in the data. this
    is required for some of the timeseries prediction.
    PARAMETERS
    ----------
    df: Pandas DataFrame
    feature: column name of smooth
    time_window: time window to aggregate over, use convention used in Pandas
                 reindex or rolling:http://pandas.pydata.org/pandas-docs/stable/timeseries.html#offset-aliases
    RETURNS
    -------
    df: Pandas DataFrame
    '''
    temp = pd.DataFrame(df[feature].shift(1))
    mean = temp.rolling(time_window,min_periods=1).mean()
    mean = mean.fillna(temp.bfill())
    std = temp.rolling(time_window,min_periods=1).std()
    std = std.fillna(temp.bfill())
    combined = (mean - 1 * std)
    result = np.where(temp < combined,None, temp)
    col_name = feature+'_old'
    df[col_name] = df[feature]
    df[feature] = result
    df[feature] = df[feature].fillna(method='ffill')
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
    df['power_all'] = ( df['power_1'] +df['power_2']+df['power_3'] ) * 5./12
    return df

if __name__=='__main__':
    project_name, smooth = sys.argv[1],sys.argv[2]
    project_name = 'project_5526_2'
    # read in data
    print 'reading clean data...'
    filename = '{}_clean.csv'.format(project_name)
    path = '../capstone_data/Azimuth/clean/{}'.format(filename)
    df = get_clean_data(path, 't')
    df = get_relay_start(df)
    df = calculate_power(df)
    if smooth == 'y':
        df = outage_smoothing(df, 'power_all', (2*24))

    # created shifted features
    print 'creating shifted features...'
    df = shift_features(df, ['load_v1rms','load_v2rms','load_v3rms',
                                       'load_i1rms','load_i2rms','laod_i3rms',
                                   'relay_est'],[1,2,3,4])

    load_feature_agg = {'H':{'std':['load_v1rms','load_v2rms','load_v3rms',
                                'load_i1rms','load_i2rms','laod_i3rms']},
                        '7D':{'sum':['relay_est','begin_gen']}}
    print 'creating time windowed aggregate load features'
    df = create_time_aggregates(df, load_feature_agg)

    # dummify categorical data
    df2 = create_dummies(df, ['month','dayofweek','hour'])

    print 'adding weather data...'
    weather = get_weather_data()

    weather_feature_agg = {'D':{'mean':['T','v3']},
                           '7D':{'sum':['liq_precip','vap_precip'],
                                'mean': ['T']},
                           '30D':{'sum':['liq_precip','vap_precip']}}

    print 'creating time windowed aggregate weather features'
    weather = create_time_aggregates(weather, weather_feature_agg)

    df3 = add_weather_data(df2, weather)
    print 'writing to csv...'
    filelocation='../capstone_data/Azimuth/clean/{}_featurized.csv'.format(project_name)
    df3.to_csv(filelocation)


    # drop continuous outage points
    # df2 = df[~((df['relay_est']==1)&(df['relay_est-1']==1))]
