# -*- coding: utf-8 -*-
"""
Created on Wed Aug 15 15:38:50 2018

@author: lyaa
"""

import sys
sys.path.insert(0, '../../bosch_helper')
from bosch_helper import *


#%% Prepare parameters
param_grid = {'max_depth': [14], 
              'eta': [0.025, 0.03, 0.035],
              'silent': [1],
              'objective': ['binary:logistic'],
              'nthread': [16],
              'lambda': [3.5, 4, 4.5],
              'alpha': [0, 0.25], 
              'subsample': [0.85, 0.9, 9.5],
              'min_child_weight': [4.5, 5, 5.5],
              'booster': ['gbtree', 'dart'],
              'base_score': [0.0058], 
              'colsample_bytree': [0.5, 0.55, 0.6, 0.65]}

param_list = list(ParameterSampler(param_grid, 
    n_iter=100, random_state=285749))

#%% Load data
x = pd.read_hdf('numeric_b1_b7_nf149.hdf', 'numeric')
y_train = pd.read_hdf('numeric_b1_b7_nf149.hdf', 'y_train')

x = x.iloc[:, :30]
x_train = x.loc['train']
x_test = x.loc['test']

#%% I/O to the function
# Input: x_train, y_train, param, random_seed
# 
random_state = 42
seed = np.random.randint(10000000)

#%% 
np.random.seed(random_state)
param = param_list[0]
seed = np.random.randint(10000000)

cv_results, clfs, running_time = \
    cross_val_predict_skf_rm_xgb(params, x_train, y_train, 
    num_boost_round=10, 
    n_splits=5, 
    n_repeats=3, 
    random_state=seed, 
    verbose_eval=True)
results = {'clfs': clfs, 'cv_results': cv_results, 'running_time': running_time}
save_pickle(results, 'results_hpopt_test_1.pickle')

#%% 
import json

with open('dict.txt', 'w') as file:
    file.write(json.dumps(param_list[0]))