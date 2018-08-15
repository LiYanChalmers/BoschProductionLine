# -*- coding: utf-8 -*-
"""
Template for CV parameter search
Tasks:
    1. CV
    2. Train model
    3. Predict on test set
    4. Save 
        a. CV results
        b. models trained in CV
        c. model trained on the whole train set
        d. predictions on test set

To-do:
    1. Use models in CV to predict on test set, and save the predictions
        a. Rewrite the CV function
        b. Overhead of prediction should be small
        c. RAM requirement should be small if #columns is not too large
        d. In some cases, may need many columns, RAM requirement may be high.
            So not implementing this idea now.
"""

import sys
sys.path.insert(0, 'bosch_helper')
from bosch_helper import *


#%% Set parameter
param_id = 94
random_state = 655944
param = {'subsample': 0.9, 'silent': 1, 'objective': 'binary:logistic', 'nthread': 20, 'min_child_weight': 5, 'max_depth': 2, 'lambda': 3.5, 'eta': 0.03, 'colsample_bytree': 0.6, 'booster': 'gbtree', 'base_score': 0.0058, 'alpha': 0}
np.random.seed(random_state)

#%% Load data
x = pd.read_hdf('numeric_b1_b7_nf149.hdf', 'numeric')
x = x.iloc[:, :30]
y_train = pd.read_hdf('numeric_b1_b7_nf149.hdf', 'y_train')
x_train = x.loc['train']
x_test = x.loc['test']

#%%
cv_results, clfs, running_time = \
    cross_val_predict_skf_rm_xgb(param, x_train, y_train, 
    num_boost_round=2,
    n_splits=2,
    n_repeats=2,
    random_state=np.random.randint(10**6), 
    verbose_eval=True)
results = {'clfs_cv': clfs, 'results_cv': cv_results, 'running_time_cv': running_time}

#%% Train on model
dtrain = xgb.DMatrix(x_train, label=y_train)
param['seed'] = np.random.randint(10**6)
clf = xgb.train(param, dtrain, 
    num_boost_round=2,
    feval=mcc_eval, evals=[(dtrain, 'train')])
y_train_pred = clf.predict(dtrain)

# Find best threshold 
thresholds = np.linspace(0.01, 0.99, 2)
mcc = np.array([matthews_corrcoef(y_train, y_train_pred>thr) for thr in thresholds])
best_threshold = thresholds[mcc.argmax()]

results['best_threshold_train'] = best_threshold
results['mcc_max_train'] = mcc.max()
results['clf_train'] = clf

#%% Predict on test set
dtest = xgb.DMatrix(x_test)
y_test_pred = clf.predict(dtest)
y_test_pred_int = (y_test_pred>best_threshold).astype(int)

sub = pd.read_csv("sample_submission.csv.zip", index_col=0)
sub["Response"] = y_test_pred_int
sub.to_csv("submission_hpopt_{}.csv.gz".format(param_id), compression="gzip")

results['y_test_pred_prob'] = y_test_pred
results['y_test_pred_int'] = y_test_pred_int

save_pickle(results, 'results_hpopt_{}.pickle'.format(param_id))