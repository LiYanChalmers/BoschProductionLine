# -*- coding: utf-8 -*-
"""
CV of station features on cluster
"""

import sys
sys.path.insert(0, 'bosch_helper')
from bosch_helper import *

#%% Load data
x = pd.read_hdf('numeric_b1_b7_nf149.hdf', 'numeric')
y_train = pd.read_hdf('numeric_b1_b7_nf149.hdf', 'y_train')

time_station = pd.read_hdf('time_station.hdf', 'time_station')

x = x.join(time_station)

x_train = x.loc['train']
x_train = x_train.iloc[:, :30]
x_test = x.loc['test']
x_test = x_test.iloc[:, :30]

#%% CV
params = {'max_depth':4, 'eta':0.03, 'silent':1, 'objective':'binary:logistic', 'nthread':20,
         'lambda':4, 'subsample':0.9, 'min_child_weight':5, 'booster':'gbtree', 'alpha':0,
         'base_score':0.0058, 'colsample_bytree':0.6}

cv_results, clfs, running_time = \
    cross_val_predict_skf_rm_xgb(params, x_train, y_train, 
    num_boost_round=2, n_splits=2, 
    n_repeats=2, random_state=5870577, 
    verbose_eval=True)

results = {'clfs_cv': clfs, 
           'results_cv': cv_results, 
           'running_time_cv': running_time}

#%% CV results
cv_train_mean = cv_results['train'].mean(axis=1)
cv_train_std = cv_results['train'].std(axis=1)
cv_test_mean = cv_results['test'].mean(axis=1)
cv_test_std = cv_results['test'].std(axis=1)

#plt.figure(figsize=(14, 7))
#plt.plot(np.arange(len(cv_train_mean)), cv_train_mean)
#plt.fill_between(np.arange(len(cv_train_mean)), cv_train_mean-cv_train_std, cv_train_mean+cv_train_std, alpha=0.5)
#plt.plot(np.arange(len(cv_train_mean)), cv_test_mean)
#plt.fill_between(np.arange(len(cv_test_mean)), cv_test_mean-cv_test_std, cv_test_mean+cv_test_std, alpha=0.5)
#plt.legend(['train', 'test'])

#%% Train data model
dtrain = xgb.DMatrix(x_train, label=y_train)
params['seed'] = 587359
clf = xgb.train(params, dtrain, 
    num_boost_round=2,
    feval=mcc_eval, 
    evals=[(dtrain, 'train')])

y_train_pred = clf.predict(dtrain)

# Find best threshold 
thresholds = np.linspace(0.01, 0.99, 4)
mcc = np.array([matthews_corrcoef(y_train, y_train_pred>thr) 
    for thr in thresholds])
#plt.plot(thresholds, mcc)
best_threshold = thresholds[mcc.argmax()]

print('Optimal MCC = {:.3f}'.format(mcc.max()))
print('Optimal threshold = {:.3f}'.format(best_threshold))

results['best_threshold_train'] = best_threshold
results['mcc_max_train'] = mcc.max()
results['clf_train'] = clf

#%% Predict on test data
dtest = xgb.DMatrix(x_test)
y_test_pred = clf.predict(dtest)
y_test_pred_int = (y_test_pred>best_threshold).astype(int)

sub = pd.read_csv("sample_submission.csv.zip", index_col=0)
sub["Response"] = y_test_pred_int
sub.to_csv("benchmark_8_submission_cv_6_station.csv.gz", compression="gzip")

save_pickle(results, 'results_benchmark_8_cv_6_station.pickle')