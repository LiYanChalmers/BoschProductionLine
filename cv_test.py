import os
if os.name=='nt':
    mingw_path = 'C:\\Program Files\\mingw-w64\\x86_64-8.1.0-posix-seh-rt_v6-rev0\\mingw64\\bin'
    os.environ['PATH'] = mingw_path + ';' + os.environ['PATH']

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import matthews_corrcoef, roc_auc_score
from sklearn.cross_validation import cross_val_score
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold
from sklearn.model_selection import cross_validate, cross_val_predict, StratifiedKFold
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from sys import getsizeof
import time
import gc



import pickle

def save_pickle(x, filename):
    with open(filename, 'wb') as handle:
        pickle.dump(x, handle, protocol=pickle.HIGHEST_PROTOCOL)

def read_pickle(filename):
    with open(filename, 'rb') as handle:
        x = pickle.load(handle)
    return x
	


def cross_val_predict_skf_rm(clf, x_train, y_train, score_callable, n_splits=3, 
                           n_repeats=2, random_state=42, verbose=False):
    '''
    Stratified KFold CV with repeated models
    '''
    y_pred_all = []
    scores = []
    n_trees = []

    skf = StratifiedKFold(n_splits=n_splits, random_state=0, shuffle=True)
    np.random.seed(random_state)

    for m in range(n_repeats):
        y_pred = []
        print('Repeat {}'.format(m))
        for n, (train_index, test_index) in enumerate(skf.split(x_train, y_train)):
            x_train_tmp, x_test_tmp = x_train[train_index], x_train[test_index]
            y_train_tmp, y_test_tmp = y_train[train_index], y_train[test_index]

            clf.random_state = np.random.randint(10000000)
            # print(clf.random_state)

            clf.fit(x_train_tmp, y_train_tmp, 
                    eval_set=[(x_test_tmp, y_test_tmp)], 
                    eval_metric='auc', early_stopping_rounds=10,
                    verbose=verbose)
            n_trees.append(clf.best_ntree_limit)
            y_pred_tmp = clf.predict_proba(x_test_tmp)[:, 1]
            y_pred.append(y_pred_tmp)
            scores.append(score_callable(y_test_tmp, y_pred_tmp))
            print('Split {}, score = {:.3f}, n_best_trees = {}'.format(n, 
                scores[m*n_repeats+n], clf.best_ntree_limit))
        y_pred_all.append(np.concatenate(y_pred).reshape((-1, 1)))
        
    y_pred_all = np.concatenate(y_pred_all, axis=1)
    print('Score mean = {:.3f}, std = {:.3f}'.format(np.mean(scores), np.std(scores)))
    
    return y_pred_all, scores, n_trees
	
	
y_train = read_pickle('y_train.pickle')

x_train = read_pickle('x_train_numeric_date_0.pickle')

n_estimators = 100
clf = XGBClassifier(max_depth=9, n_estimators=n_estimators, 
                    base_score=0.0058, n_jobs=-1, colsample_bytree=0.6,
                    min_child_weight=5, subsample=0.9,  
                    reg_lambda=4, silent=False, learning_rate=0.03)
					
y_pred, scores, n_trees = cross_val_predict_skf_rm(clf, x_train, y_train, 
                                         roc_auc_score, n_splits=5, 
                                         n_repeats=3, random_state=42, verbose=True)