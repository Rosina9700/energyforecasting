import warnings
import itertools
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from datetime import datetime
import sys

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from backports import weakref
from datetime import datetime
from baseline_models import Baseline_average, Baseline_previous
from data_wrangling import Results_data, Data_preparation

np.random.seed(7)

def shift_features(df, feature, deltas):
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
    col_names = [feature]
    shifted_dfs = [df]
    for d in deltas:
        shifted = df[feature].shift(d)
        name = '{}-{}'.format(feature,d)
        col_names.append(name)
        shifted_dfs.append(shifted)
    new_df = pd.concat(shifted_dfs,axis=1)
    print col_names
    new_df.columns = col_names
    new_df.dropna(inplace=True)
    return new_df

def scale_data(y_train, y_test):
    scaler = MinMaxScaler(feature_range=(0, 1)).fit(y_train[:,0].reshape(len(y_train),1))
    y_train_scaled = scaler.transform(y_train)
    y_test_scaled = scaler.transform(y_test)
    return y_train_scaled, y_test_scaled, scaler


def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back)]
		dataX.append(a)
		dataY.append(dataset[i + look_back])
	return np.array(dataX), np.array(dataY)

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
    dataset = y.ix[:,0]

    cv_folds = 5
    # look_back = 4
    # deltas = [i+1 for i in range(look_back)]
    # y = shift_features(y, 'power_all', deltas)

    # split into train and test sets
    train, test = dataset[0:-2*season], dataset[-2*season:]
    # reshape into X=t and Y=t+1
    look_backs = [3,4]
    epochs = [100,200]
    batch_sizes = [1]
    nodes = [3,4]
    combo = []
    combo_rmse = []
    for look_back in look_backs:
        for epoch in epochs:
            for batch_size in batch_sizes:
                for node in nodes:
                    combo.append((look_back, epoch, batch_size, node))
                    # look_back = 3
                    length = len(train)-1
                    chunks = min(cv_folds, length/2)
                    chunk_size = (length/2)/chunks
                    rmses=[]
                    for i in xrange(chunks+1):
                        end_index = (length/2) + (i)*chunk_size
                        print 'data length: {} chunks size: {}'.format(length, end_index)
                        # normalize the dataset
                        scaler = MinMaxScaler(feature_range=(0, 1))
                        scaler = scaler.fit(train[:end_index])
                        X, Y = create_dataset(train[:end_index+1], look_back)
                        trainX, trainY = scaler.transform(X[:-1]), scaler.transform(Y[:-1])
                        valX = scaler.transform(X[-1])
                        valY = Y[-1]
                        trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
                        print valX.shape
                        valX = np.reshape(valX, (1, valX.shape[0], 1))
                        # create and fit the LSTM network
                        # batch_size = 1
                        model = Sequential()
                        model.add(LSTM(node, batch_input_shape=(batch_size, look_back, 1), stateful=True, return_sequences=True))
                        model.add(LSTM(node, batch_input_shape=(batch_size, look_back, 1), stateful=True))
                        model.add(Dense(1))
                        model.compile(loss='mean_squared_error', optimizer='adam')
                        for i in range(epoch):
                        	model.fit(trainX, trainY, epochs=1, batch_size=batch_size, verbose=2, shuffle=False)
                        	# model.reset_states()
                        valPredict = model.predict(valX, batch_size=batch_size)
                        valPredict = scaler.inverse_transform(valPredict)
                        rmse = np.sqrt((valPredict-valY)**2)
                        rmses.append(rmse)
                    score = np.asarray(rmses).mean()
                    combo_rmse.append(score)
                    print combo
                    print score
    ind_best = np.array(combo_rmse).argmin()
    best_combo = combo[ind_best]
    best_combo_rmse = combo_rmse[ind_best]

    look_back = best_combo[0]
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler = scaler.fit(dataset[:-2*season])
    X, Y = create_dataset(dataset, look_back)
    trainX, trainY = scaler.transform(X[:-2*season]), scaler.transform(Y[:-2*season])
    testX, testY = scaler.transform(X[-2*season:]), scaler.transform(Y[-2*season:])

    # trainX, trainY = create_dataset(train, look_back)
    # testX, testY = create_dataset(test, look_back)
    # reshape input to be [samples, time steps, features]
    trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
    testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1))

    # create and fit the LSTM network
    batch_size = best_combo[2]
    model = Sequential()
    model.add(LSTM(best_combo[3], batch_input_shape=(batch_size, look_back, 1), stateful=True, return_sequences=True))
    model.add(LSTM(best_combo[3], batch_input_shape=(batch_size, look_back, 1), stateful=True))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    for i in range(best_combo[1]):
    	model.fit(trainX, trainY, epochs=1, batch_size=batch_size, verbose=2, shuffle=False)
    	# model.reset_states()

    # make predictions
    trainPredict = model.predict(trainX, batch_size=batch_size)
    model.reset_states()
    testPredict = model.predict(testX, batch_size=batch_size)
    # invert predictions
    trainPredict = scaler.inverse_transform(trainPredict)
    trainY = scaler.inverse_transform([trainY])
    testPredict = scaler.inverse_transform(testPredict)
    testY = scaler.inverse_transform([testY])
    # calculate root mean squared error
    print 'Cross val score {}'.format(best_combo_rmse)
    trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
    print('Train Score: %.2f RMSE' % (trainScore))
    testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
    print('Test Score: %.2f RMSE' % (testScore))



        # target, timesteps = get_ready_for_lstm(df, feature='power_all', deltas=[1,2,3,4],freq='H')
        # y_train = y[:-24]
        # y_test = y[-24:]
        # cv_folds = 10
        # #
        # print '\nbaseline - previous...'
        # b_previous = Baseline_previous()
        # b1_train_rmse, model = baseline_cross_val_score(b_previous, y_train, cv_folds)
        # forecast, b1_test_rmse, model = baseline_rolling_predictions(b_previous, y,len(y_train)-24,24)
        # print 'Baseline-previous train RMSE {}'.format(b1_train_rmse)
        # print 'Baseline-previous test RMSE {}'.format(b1_test_rmse)
        # #
        # print 'baseline - averages....'
        # b_average = Baseline_average()
        # b2_train_rmse, model = baseline_cross_val_score(b_average, y_train, cv_folds)
        # forecast, b2_test_rmse, model = baseline_rolling_predictions(b_average, y,len(y_train)-24,24)
        # print 'Baseline-averages train RMSE {}'.format(b2_train_rmse)
        # print 'Baseline-averages test RMSE {}'.format(b2_test_rmse)
        #
        # print '\nfind best sarima...'
        # y_train = y[:-24]
        # y_test = y[-24:]
        # p = range(0,5)
        # q = range(1,3)
        # d = range(0,2)
        # # p = range(1,2)
        # # q = range(0,1)
        # # d = range(0,1)
        # params = (p,d,q)
        # model, s_train_rmse = find_best_sarima(y_train,params,24, k=cv_folds)
        # best_params = model.specification
        # params = (best_params['order'],best_params['seasonal_order'])
        # test_forecast, s_test_rmse, model = rolling_predictions_sarima(y,len(y_train)-24,24,params)
        # print('SARIMA{}x{}{} - AIC:{}'.format(best_params['order'], best_params['seasonal_order'],
        #                                  best_params['seasonal_periods'],model.aic))
        # print 'Training cross validation RMSE: {}'.format(s_train_rmse)
        # print'Test cross validation RMSE {}'.format(s_test_rmse)
        # # #
        # now = datetime.now().strftime('%m_%d_%H_%M_%S')
        # filename = 'output_{}_{}.txt'.format(project_name,now)
        # test = 1
        # train = 0
        # with open(filename, "w") as text_file:
        #     train_results = '{},{},{},{},{},{}'.format(project_name,train,b1_train_rmse,b2_train_rmse,s_train_rmse,best_params)
        #     test_results = '{},{},{},{},{},{}'.format(project_name,test,b1_test_rmse,b2_test_rmse,s_test_rmse,best_params)
        #     string_to_write = 'project,test,baseline_previous,baseline_averages,sarima,sarima_params\n{}\n{}'.format(train_results,test_results)
        #     text_file.write(string_to_write)
