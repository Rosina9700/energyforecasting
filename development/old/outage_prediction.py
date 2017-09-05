import pandas as pd
import plotting as mp
import numpy as np
import sys
from operator import itemgetter
from collections import defaultdict
from datetime import datetime

from sklearn.metrics import recall_score, precision_score, accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def get_featurized_data(project_name):
    df = pd.read_csv('../capstone_data/Azimuth/clean/{}_featurized.csv'.format(project_name))
    df['t'] = pd.to_datetime(df['t'], format='%Y-%m-%d %H:%M:%S')
    df.set_index('t',inplace=True)
    return df

def remove_continuous_outage_points(df):
    return df[~((df['begin_gen']==0)&(df['relay_est']==1))]

def prepare_dataset(df):
    df = remove_continuous_outage_points(df)
    cols_to_drop = ['project_id','relay', 'datetime_hr',
                'load_v1rms','load_v2rms',"load_v3rms","load_i1rms","load_i2rms","laod_i3rms",
                'year','day','relay_est','t_diff','t_diff-1','t_diff-2','t_diff-3','t_diff-4',
                'data_issue','begin_gen','relay_est-1','lat','lon','cumulated hours',
                'd1','d2','d3','z0','h1','h2','h3']
    y = df['begin_gen']
    X = df.drop(cols_to_drop, axis=1)
    return X, y

class Baseline_model(object):
    def __init__(self):
        self.baseline_probs = None

    def train(self, X_train, y_train):
        X_train['hour'] = X_train.index.hour
        self.baseline_prob = X_train.groupby('hour')['begin_gen'].mean()
        X_train.drop('hour', axis=1, inplace=True)
        return self

    def predict(self, X):
        thresholds = X.index.hour.values.apply(lambda x: self.baseline_prob[x])
        random_guess = np.random.uniform(0,1,len(thresholds))
        predictions = (random_guess < thresholds)
        return predictions

    def evaluate(self, X, y):
        scores = {'recall':[], 'precision':[],'f1':[],'accuracy':[]}
        for i in xrange(10):
            predictions = self.predict(X)
            scores['recall'].append(recall_score(y, predictions))
            scores['precision'].append(precision_score(y, predictions))
            scores['f1'].append(f1_score(y, predictions))
            scores['accuracy'].append(accuracy_score(y, predictions))
        for key, item in scores.iteritems():
            scores[key] = np.asarray(item).mean()
        return scores

def over_under_sample(X_train, y_train, under_factor, over_factor):
    majority_ind = y_train[y_train==0].index
    random_majority = np.random.choice(majority_ind, len(majority_ind)/under_factor,replace=False)

    minority_ind = y_train[y_train==1].index
    random_minority = np.random.choice(minority_ind, len(minority_ind)*over_factor,replace=True)

    X_train2 = X_train.loc[random_majority].append(X_train.loc[random_minority])
    y_train2 = y_train.loc[random_majority].append(y_train.loc[random_minority])
    return X_train2, y_train2

def classifier(classifier, X_train, y_train):
    model = classifier
    model = model.fit(X_train, y_train)
    return model

def classifier_evaluation(model, X, y):
    y_pred = model.predict(X)
    scores = defaultdict()
    scores['recall'] = recall_score(y, y_pred)
    scores['precision'] = precision_score(y, y_pred)
    scores['f1'] = f1_score(y, y_pred)
    scores['accuracy'] = accuracy_score(y, y_pred)
    return scores

def get_feature_importance(model, X_train):
    feat_importance = zip(model.feature_importances_,X_train.columns)
    feat_importance = sorted(feat_importance, key = lambda x: x[0],reverse=True)
    return feat_importance


if __name__ == '__main__':
    project = 'project_1074'
    print '\n\n{} at {}'.format(project,datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    print 'getting data...'
    df = get_featurized_data(project)
    X, y = prepare_dataset(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=10)

    print 'Random forest without addressing class imbalance...'
    rfc_classifier = classifier(RandomForestClassifier(oob_score=True, n_jobs=-1),X_train, y_train)
    train_eval = classifier_evaluation(rfc_classifier, X_train, y_train)
    test_eval = classifier_evaluation(rfc_classifier, X_test, y_test)
    feature_importances = get_feature_importance(rfc_classifier, X_train)
    print '\nTop 20 features:'
    for f in feature_importances[:10]:
        print f
    print '\nTraining evaluation:'
    for key, value in train_eval.iteritems():
        print '{}: {}'.format(key, value)
    print '\nTesting evaluation:'
    for key, value in test_eval.iteritems():
        print '{}: {}'.format(key, value)

    print '\n\nAttempt to address class imbalance'
    X_train2, y_train2 = over_under_sample(X_train, y_train, 5, 5)
    rfc_classifier = classifier(RandomForestClassifier(oob_score=True, n_jobs=-1),X_train2, y_train2)
    train_eval = classifier_evaluation(rfc_classifier, X_train2, y_train2)
    test_eval = classifier_evaluation(rfc_classifier, X_test, y_test)
    feature_importances = get_feature_importance(rfc_classifier, X_train2)
    print '\nTop 20 features:'
    for f in feature_importances[:10]:
        print f
    print '\nTraining evaluation:'
    for key, value in train_eval.iteritems():
        print '{}: {}'.format(key, value)
    print '\nTesting evaluation:'
    for key, value in test_eval.iteritems():
        print '{}: {}'.format(key, value)
