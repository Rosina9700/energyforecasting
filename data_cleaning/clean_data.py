import pandas as pd

def get_raw_data(path, time_col):
    df = pd.read_csv(path)
    df.dropna(inplace=True)
    df['t'] = pd.to_datetime(df[time_col], format='%Y-%m-%d %H:%M:%S')
    df.sort_values('t',inplace=True)
    df = df.set_index('t')
    df['year'] = df.index.year
    df['month'] = df.index.month
    df['day'] = df.index.day
    df['dayofweek'] = df.index.dayofweek
    df['hour'] = df.index.hour
    return df
