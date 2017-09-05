import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

def line_plot_predictions(project_name, forecasts, true):
    x_axis = true.index.values
    fig, axes = plt.subplots(len(forecasts), sharex=True, sharey=True, figsize=(15,6))
    title = 'Model performance on {}'.format(project_name)
    fig.suptitle(title)
    fig.text(0.5, 0.02, 'Time', ha='center')
    fig.text(0.04, 0.5, 'Energy demand (Wh)', va='center', rotation='vertical')
    # plt.figure(figsize=(20,8))
    # ax[0].plot(x_axis, true.ix[:,0].values, label='measured')
    color = ['m','b','g','c','y']
    counter = 0
    for ax, f in zip(axes,forecasts):
        ax.plot(x_axis, true.ix[:,0].values,label='measured',color='r' ,alpha=0.75)
        ax.plot(x_axis, f[1], '--',label=f[0], color=color[counter], alpha=0.75)
        ax.fill_between(x_axis, true.iloc[:,-1]*true.ix[:,0], alpha=0.4, label='Weekday')
        ax.legend(loc=1)
        # ax.set_xlabel('Time')
        # ax.set_ylabel('Energy Demand (Wh)')
        counter += 1
    plt.show()
    pass

def scatter_plot_predictions(forecasts, true):
    x_axis = true.index.values
    fig, axes = plt.subplots(1,len(forecasts), sharex=True, sharey=True, figsize=(30,5))
    fig.suptitle('Model performance on test data')
    fig.text(0.5, 0.02, 'Measured energy demand values', ha='center')
    fig.text(0.04, 0.5, 'Forecasted energy demand (Wh)', va='center', rotation='vertical')
    # plt.figure(figsize=(20,8))
    # ax[0].plot(x_axis, true.ix[:,0].values, label='measured')
    color = ['m','b','g','o']
    counter = 0
    for ax, f in zip(axes,forecasts):
        ax.scatter(true.ix[:,0].values, f[1],label=f[0],color=color[counter] ,alpha=0.75)
        ax.legend(loc=1)
        # ax.set_xlabel('Time')
        # ax.set_ylabel('Energy Demand (Wh)')
        counter += 1
    pass

def cross_val_scores(df,var=False):
    cross_val_scores = {}
    cross_val_scores['baseline1'] = df.iloc[0]['baseline_previous']
    cross_val_scores['baseline2'] = df.iloc[0]['baseline_averages']
    df1 = df[df['beta_var']==0]
    cross_val_scores['sarima'] = df1[df1['model']=='sarima']['sarimax'].values[0]
    cross_val_scores['sarimaX'] = df1[df1['model']=='sarimaX']['sarimax'].values[0]
    df2 = df[df['beta_var']==1]
    if var==True:
        cross_val_scores['sarimaX_variable'] =  df2[df2['model']=='sarimaX']['sarimax'].values[0]
    results = OrderedDict(sorted(cross_val_scores.items(), key=lambda t: t[0]))
    return results

def plot_cross_val_score(df):
    results = cross_val_scores(df)
    values = [v/1000. for v in results.values()]
    plt.figure(figsize=(8,6))
    plt.bar(range(len(results)),values, color='b', alpha=0.5,align='center')
    plt.xticks(range(len(results)), results.keys(),rotation=45)
    plt.ylabel('Cross validation RMSE [kWh]')
    plt.title(df.iloc[0]['project'])
    plt.show()
    pass
