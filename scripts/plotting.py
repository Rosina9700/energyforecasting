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
    '''
    Function plots the forecasts for the different models against the
    measured data
    parameters
    -----------
    project_name: String
    forecasting: List of lists
    true: Numpy array
    RETURNS
    --------
    Matplotlib plot
    '''
    x_axis = true.index.values
    fig, axes = plt.subplots(len(forecasts), sharex=True, sharey=True, figsize=(15,8))
    title = 'Model performance on {}'.format(project_name)
    fig.suptitle(title)
    fig.text(0.5, 0.01, 'Time', ha='center')
    fig.text(0.00, 0.5, 'Energy demand (Wh)', va='center', rotation='vertical')
    plt.figure(figsize=(20,8))
    color = ['m','b','g','r','y']
    counter = 0
    for ax, f in zip(axes,forecasts):
        ax.plot(x_axis, true.ix[:,0].values,label='measured',color='k' ,alpha=0.75)
        ax.plot(x_axis, f[1], '--',label=f[0], color=color[counter], alpha=0.75)
        ax.fill_between(x_axis, true.iloc[:,-1]*true.ix[:,0], alpha=0.4, label='Weekday')
        ax.set_ylim(ymin=np.min(true.ix[:,0].values)/1.2)
        ax.legend(loc=2)
        counter += 1
    plt.show()
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

def plot_norm_cross_val_score(scores):
    font_size=30
    width =  0.6
    scaled_rmses = np.asarray([s[1].values() for s in scores])
    fig, ax = plt.subplots(figsize=(10,8))
    ax.bar(range(4),scaled_rmses.mean(axis=0),0.5, color='b', alpha=0.5, align='center')
    ax.set_xticks(range(4))
    ax.set_xticklabels(scores[0][1].keys(),rotation=45,fontsize=font_size)
    ax.tick_params(labelsize=20)
    ax.set_ylabel('Relative RMSE', fontsize=font_size-5)
    ax.set_title('Cross validation RMSE comparison',fontsize=font_size)
    plt.show()
    pass

def plot_all_cross_val_score(results, project_type):
    fig, axes = plt.subplots(2,3, figsize=(28,15),sharex=True)
    font_size=30
    width =  0.6
    for df, ax in zip(results, axes.ravel()):
        scores = cross_val_scores(df)
        values = [v/1000. for v in scores.values()]
        ax.bar(range(len(scores)),values, width, color='b', alpha=0.5,align='center')
        ax.set_xticks(range(len(scores)))
        ax.set_xticklabels(scores.keys(),rotation=45,fontsize=font_size)
        ax.tick_params(labelsize=25)
        ax.set_ylabel('Cross validation RMSE [kWh]', fontsize=font_size-5)
        ax.set_title(project_type[df['project'].values[0]],fontsize=font_size+5)
    plt.tight_layout()
    plt.show()
    pass
