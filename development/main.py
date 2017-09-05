import warnings
import pandas as pd
import csv
from datetime import datetime
from baseline_models import Baseline_average, Baseline_previous, baseline_rolling_predictions, baseline_cross_val_score
from data_wrangling import Results_data, Data_preparation
from sarima_models import Sarima_predictions, find_best_sarima, grid_search_sarima
import sys

if __name__== '__main__':
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
    cv_folds = 5
    model_names = ['sarima','sarimax','sarimax_v']
    combination = [0,1]

    y_train = y[:-2*season]
    y_test = y[-2*season:]

    print '\nbaseline - previous...'
    b_previous = Baseline_previous()
    b1_train_rmse, model = baseline_cross_val_score(b_previous, pd.DataFrame(y_train.ix[:,0]), cv_folds, window=1, season=season)
    forecast, b1_test_rmse, model = baseline_rolling_predictions(b_previous, pd.DataFrame(y.ix[:,0]),len(y_train),2*season)
    print 'Baseline-previous train RMSE {}'.format(b1_train_rmse)
    print 'Baseline-previous test RMSE {}'.format(b1_test_rmse)

    print 'baseline - averages....'
    b_average = Baseline_average(freq=f)
    b2_train_rmse, model = baseline_cross_val_score(b_average, pd.DataFrame(y_train.ix[:,0]), cv_folds, window=1, season=season)
    forecast, b2_test_rmse, model = baseline_rolling_predictions(b_average, pd.DataFrame(y.ix[:,0]),len(y_train),2*season)
    print 'Baseline-averages train RMSE {}'.format(b2_train_rmse)
    print 'Baseline-averages test RMSE {}'.format(b2_test_rmse)

    print '\nfind best sarima...'
    y_train = y[:-2*season]
    y_test = y[-2*season:]
    if f == 'D':
        p = range(0,4)
    else:
        p = range(0,3)
    q = range(1,3)
    d = range(0,2)
    params = (p,d,q)
    best_models = find_best_sarima(y_train, params, season, combination, k=cv_folds)

    to_print = ['project;model;test;baseline_previous;baseline_averages;sarimax;sarimax_aic;sarimax_params']
    test = 1
    train = 0
    for key, results in best_models.iteritems():
        model = results[0]
        train_rmse = results[1]
        best_params = model.specification
        params = (best_params['order'],best_params['seasonal_order'])
        print('SARIMA{}x{}{} - AIC:{}'.format(params[0], params[1],
                                         best_params['seasonal_periods'],model.aic))
        sp = Sarima_predictions(params, model_names.index(key))
        results = sp.rolling_predictions_sarima(y,len(y_train),2*season)
        test_rmse = results[1]
        print '{} training cross validation RMSE: {}'.format(key, train_rmse)
        print'{} test RMSE {}'.format(key, test_rmse)
        train_results = '{};{};{};{};{};{};{};{}'.format(project_name, key, train, b1_train_rmse, b2_train_rmse, train_rmse, model.aic, best_params)
        test_results = '{};{};{};{};{};{};{};{}'.format(project_name, key, test, b1_test_rmse, b2_test_rmse, test_rmse, results[2].aic, best_params)
        to_print.append(train_results)
        to_print.append(test_results)

    now = datetime.now().strftime('%m_%d_%H_%M_%S')
    filename = 'output_{}_{}.csv'.format(project_name,now)
    with open(filename, "wb") as file:
        for r in to_print:
            file.write(r)
            file.write('\n')
