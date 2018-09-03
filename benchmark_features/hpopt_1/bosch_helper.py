import os
if os.name=='nt':
    try:
        mingw_path = 'C:\\Program Files\\mingw-w64\\x86_64-8.1.0-posix-seh-rt_v6-rev0\\mingw64\\bin'
        os.environ['PATH'] = mingw_path + ';' + os.environ['PATH']
    except:
        pass

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
import xgboost as xgb
from sklearn.metrics import matthews_corrcoef, roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold
from sklearn.model_selection import cross_validate, cross_val_predict, StratifiedKFold
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from sys import getsizeof
import time
import gc
import tqdm

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import ParameterSampler

from scipy import fftpack

import warnings
warnings.filterwarnings("ignore")


import pickle

def save_pickle(x, filename):
    with open(filename, 'wb') as handle:
        pickle.dump(x, handle, protocol=pickle.HIGHEST_PROTOCOL)

def read_pickle(filename):
    with open(filename, 'rb') as handle:
        x = pickle.load(handle)
    return x

from numba import jit

@jit
def mcc(tp, tn, fp, fn):
    sup = tp * tn - fp * fn
    inf = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    if inf==0:
        return 0
    else:
        return sup / np.sqrt(inf)

@jit
def eval_mcc(y_true, y_prob, show=False):
    idx = np.argsort(y_prob)
    y_true_sort = y_true[idx]
    n = y_true.shape[0]
    nump = 1.0 * np.sum(y_true) # number of positive
    numn = n - nump # number of negative
    tp = nump
    tn = 0.0
    fp = numn
    fn = 0.0
    best_mcc = 0.0
    best_id = -1
    prev_proba = -1
    best_proba = -1
    mccs = np.zeros(n)
    new_mcc = 0
    for i in range(n):
        # all items with idx < i are predicted negative while others are predicted positive
        # only evaluate mcc when probability changes
        proba = y_prob[idx[i]]
        if proba != prev_proba:
            prev_proba = proba
            new_mcc = mcc(tp, tn, fp, fn)
            if new_mcc >= best_mcc:
                best_mcc = new_mcc
                best_id = i
                best_proba = proba
        mccs[i] = new_mcc
        if y_true_sort[i] == 1:
            tp -= 1.0
            fn += 1.0
        else:
            fp -= 1.0
            tn += 1.0
    if show:
        y_pred = (y_prob >= best_proba).astype(int)
        score = matthews_corrcoef(y_true, y_pred)
        print(score, best_mcc)
        plt.plot(mccs)
        return best_proba, best_mcc, y_pred
    else:
        return best_mcc
    
def mcc_eval(y_prob, dtrain):
    y_true = dtrain.get_label()
    best_mcc = eval_mcc(y_true, y_prob)
    return 'MCC', best_mcc

def cross_val_predict_rskf(clf, x_train, y_train, n_splits=3, 
                           n_repeats=2, random_state=42, verbose=False, early_stopping=10):
    '''
    Repeated stratified KFold CV, returns predictions for 
    each repeat and average score.
    n_repeats: repetitions of CV
    to disable erlay stopping, set early_stopping to None
    '''
    scores = []
    n_trees = []
    clfs = []
    running_time = []
    
    rskf = RepeatedStratifiedKFold(n_repeats=n_repeats, n_splits=n_splits, 
                                   random_state=0)
    np.random.seed(random_state)
    for n, (train_index, test_index) in enumerate(rskf.split(x_train, y_train)):
        print('Round {}'.format(n))
        start_time = time.time()
        x_train_tmp, x_test_tmp = x_train.iloc[train_index], x_train.iloc[test_index]
        y_train_tmp, y_test_tmp = y_train.iloc[train_index], y_train.iloc[test_index]

        clf.random_state = np.random.randint(10000000)
        
        if early_stopping is not None:
            clf.fit(x_train_tmp, y_train_tmp, 
                    eval_set=[(x_test_tmp, y_test_tmp)], 
                    eval_metric=mcc_eval, early_stopping_rounds=early_stopping,
                    verbose=verbose)
            scores.append(-clf.best_score)
            n_trees.append(clf.best_ntree_limit)
        else:
            clf.fit(x_train_tmp, y_train_tmp)
            scores.append(eval_mcc(y_test_tmp.values, clf.predict_proba(x_test_tmp)[:, 1]))
            n_trees.append(clf.n_estimators)
        
        clfs.append(clf)
        running_time.append(time.time()-start_time)
        print('Split {}, score = {:.3f}, best_ntree_limit = {}, total time = {:.3f} min'.format(n, scores[n], 
            n_trees[n], sum(running_time)/60))

    print('Score mean = {:.3f}, std = {:.3f}'.format(np.mean(scores), np.std(scores)))
    
    return clfs, scores, n_trees, running_time

def cross_val_predict_skf_rm(clf, x_train, y_train, n_splits=3, 
                           n_repeats=2, random_state=42, verbose=False, early_stopping=10):
    '''
    Stratified KFold CV with repeated models
    to disable erlay stopping, set early_stopping to None
    '''
    scores = []
    n_trees = []
    clfs = []
    running_time = []

    skf = StratifiedKFold(n_splits=n_splits, random_state=0, shuffle=True)
    np.random.seed(random_state)

    for m in range(n_repeats):
        print('Repeat {}'.format(m))
        for n, (train_index, test_index) in enumerate(skf.split(x_train, y_train)):
            start_time = time.time()
            x_train_tmp, x_test_tmp = x_train.iloc[train_index], x_train.iloc[test_index]
            y_train_tmp, y_test_tmp = y_train.iloc[train_index], y_train.iloc[test_index]

            clf.random_state = np.random.randint(10000000)
            # print(clf.random_state)

            if early_stopping is not None:
                clf.fit(x_train_tmp, y_train_tmp, 
                        eval_set=[(x_test_tmp, y_test_tmp)], 
                        eval_metric=mcc_eval, early_stopping_rounds=early_stopping,
                        verbose=verbose)
                scores.append(-clf.best_score)
                n_trees.append(clf.best_ntree_limit)
            else:
                clf.fit(x_train_tmp, y_train_tmp)
                scores.append(eval_mcc(y_test_tmp.values, clf.predict_proba(x_test_tmp)[:, 1]))
                n_trees.append(clf.n_estimators)
            
            clfs.append(clf)
            running_time.append(time.time() - start_time)
            print('Split {}, score = {:.3f}, n_best_trees = {}, total time = {:.3f} min'.format(n, 
                scores[m*n_repeats+n], n_trees[m*n_repeats+n], sum(running_time)/60))

    print('Score mean = {:.3f}, std = {:.3f}'.format(np.mean(scores), np.std(scores)))
    
    return clfs, scores, n_trees, running_time

def cross_val_predict_skf_rm_xgb(params, x_train, y_train, num_boost_round=3, n_splits=3, 
                           n_repeats=2, random_state=3795264, verbose_eval=False):
    '''
    Stratified KFold CV with repeated models
    Early stopping is totally disabled
    Uses xgb.cv API
    verbose_eval is the same as in xgb.train
    '''
    cv_results = {}
    clfs = {}
    running_time = {}
    
    np.random.seed(random_state)
    skf = StratifiedKFold(n_splits=n_splits, random_state=np.random.randint(10**6), shuffle=True)
    
    for m in range(n_repeats):
        for n, (train_index, test_index) in enumerate(skf.split(x_train, y_train)):
            
            start_time = time.time()
            
            # Construct DMatrix
            dtrain = xgb.DMatrix(x_train.iloc[train_index], label=y_train.iloc[train_index])
            dtest = xgb.DMatrix(x_train.iloc[test_index], label=y_train.iloc[test_index])
            
            # Placeholder for evals_result
            cv_results[m, n] = {}
            params['seed'] = np.random.randint(10**6)
            clfs[m, n] = xgb.train(params, dtrain, num_boost_round=num_boost_round,
                                   evals=[(dtrain, 'train'), (dtest, 'test')],
                                  feval=mcc_eval, maximize=True, early_stopping_rounds=None, 
                                  evals_result=cv_results[m, n], verbose_eval=verbose_eval)
        
            running_time[m, n] = time.time() - start_time
            
            print('Repeat {}, split {}, test MCC = {:.3f}, running time = {:.3f} min'.format(m, n, 
                cv_results[m, n]['test']['MCC'][-1], running_time[m, n]/60))
        
    # Post-process cv_results
    cv_results_final = {}
    for m in range(n_repeats):
        for n in range(n_splits):
            cv_results_final['train', m, n] = cv_results[m, n]['train']['MCC']
            cv_results_final['test', m, n] = cv_results[m, n]['test']['MCC']
    
    df = pd.DataFrame.from_dict(cv_results_final)
    df.index.name = 'iteration'
    df.columns.names = ['dataset', 'repeat', 'split']

    print('Score mean = {:.3f}, std = {:.3f}'.format(df['test'].iloc[-1].mean(), df['test'].iloc[-1].std()))
    
    return df, clfs, running_time

def time_difference_to_failures(x, failure_max):
    '''
    Find the average time difference to the last and next failure_max failures.
    x should have ['Response', 'time_start', 'time_end']
    '''
    
    u = x[['Response', 'time_start']].copy()
    u.columns = ['Response', 'time']
    x_start = time_difference_to_failures_helper(u, failure_max, '_start')
    
    u = x[['Response', 'time_end']].copy()
    u.columns = ['Response', 'time']
    x_end = time_difference_to_failures_helper(u, failure_max, '_end')
    
    return x_start.join(x_end)

def time_difference_to_failures_helper(x, failure_max, suffix):
    '''
    Find the mean time difference since last/next 1, 2, ..., failure_max failures
    when samples are sorted by the time column
    suffix is used for column names of the final results
    x is a DataFrame containing:
    - Both train and test data
    - Two columns: Response and a time column, 
        which is used to sort samples and calculate time differences
    '''
    
    # sort by time and Id
    x.sort_values(['time', 'Id'], inplace=True)
    x.Response.fillna(0, inplace=True)
    x.Response = x.Response.astype(np.int8)
    
    # ranking in sorted order
    x['rank_sort_time'] = np.arange(1, len(x)+1)
    
    # rank of failures
    x['rank_failure'] = x['Response']
    x['rank_failure'] = x['rank_failure'].cumsum()
    x.loc[x['Response']!=1, 'rank_failure'] = 0
    
    # the rank_failure of the 1st previous failure for each sample
    # for the first several samples without previous failures, use 0
    x['fp1'] = x['rank_failure'].shift().fillna(0).astype(np.int64)
    x['fp1'] = x['fp1'].cummax()
    x['fp1'] = x['fp1'].astype(np.int64)
    
    # the failure rank of the 2nd to failure_max previous failure for each sample
    failure_list = np.arange(2, failure_max+1)
    for f in failure_list:
        x['fp'+str(f)] = x['fp'+str(f-1)]-1
        x.loc[x['fp'+str(f)]<0, 'fp'+str(f)] = 0
        x['fp'+str(f)] = x['fp'+str(f)].astype(np.int64)
        
    # the failure rank of the 1st next failure for each sample
    # for the last several samples do not have next failure, use failure_count+1
    failure_count = sum(x['Response'])
    x['fn1'] = x['fp1'].shift(-1).fillna(failure_count)+1
    # the total number of failures
    # assign np.nan to rows whose rank_failure is larger than failure_count
    x.loc[x['fn1']>failure_count, 'fn1'] = failure_count+1
    x['fn1'] = x['fn1'].astype(np.int64)

    # the failure rank of the 2nd to failure_max (failure_max=10 here) next failure for each sample
    for f in failure_list:
        x['fn'+str(f)] = x['fn'+str(f-1)]+1
        x.loc[x['fn'+str(f)]>failure_count, 'fn'+str(f)] = failure_count+1
        x['fn'+str(f)] = x['fn'+str(f)].astype(np.int64)
        
    # a mapping from failure rank to start time of the failure
    rank_failure_to_time = x.loc[x['rank_failure']!=0, ['rank_failure', 'time']].set_index(
        'rank_failure', drop=True, inplace=False)
    rank_failure_to_time = rank_failure_to_time.to_dict()
    rank_failure_to_time = rank_failure_to_time['time']
    rank_failure_to_time[0] = np.nan
    rank_failure_to_time[failure_count+1] = np.nan
    
    # map from failure rank to time of the failure 
    for f in range(1, failure_max+1):
        x['fp{}_time'.format(f)] = x['fp'+str(f)].map(rank_failure_to_time)
        
    # map from failure rank to time of the failure 
    for f in range(1, failure_max+1):
        x['fn{}_time'.format(f)] = x['fn'+str(f)].map(rank_failure_to_time)
        
    # Calculate average of the next n failures
    ave_list = []
    for f in range(1, failure_max+1):
        ave_list.append('fn{}_time'.format(f))
        x['fn{}_time_ave{}'.format(f, suffix)] = x[ave_list].mean(axis=1) - x['time']

    # Calculate average of the previous n failures
    ave_list = []
    for f in range(1, failure_max+1):
        ave_list.append('fp{}_time'.format(f))
        x['fp{}_time_ave{}'.format(f, suffix)] = x[ave_list].mean(axis=1) - x['time']
        
    # drop auxiliary columns
    drop_list = [k for i in range(1, failure_max+1) for k in ['fp'+str(i), 'fn'+str(i)]]
    drop_list.extend([k for f in range(1, failure_max+1) for k in ['fp'+str(f)+'_time', 'fn'+str(f)+'_time']])
    drop_list.extend(['time', 'rank_sort_time', 'rank_failure', 'Response'])
    x.drop(drop_list, axis=1, inplace=True)
    
    # sort index
    x.sort_index(by='Id', axis=0, inplace=True)    
    
    return x