# -*- coding: utf-8 -*-
"""
Generate files
"""

import sys
sys.path.insert(0, '../../bosch_helper')
from bosch_helper import *

from shutil import copyfile

def change_template(src, dst, replace_lines):
    destination = open(dst, 'w')
    source = open(src, 'rU')
    for l, line in enumerate(source):
        if l in replace_lines.keys():
            destination.write(replace_lines[l])
        else:
            destination.write(line)
    source.close()
    destination.close()
    
def eol_win2unix(file):
    with open(file, 'rU') as infile:
        with open(os.path.join(file+'~'), 'w', newline='\n') as outfile:
            outfile.writelines(infile.readlines())
    os.remove(file)
    os.rename(file+'~', file)

#%% Random state
random_state = 49576
np.random.seed(random_state)

#%% Prepare parameters

# CV parameters
num_boost_round_cv = 2 # 80
n_splits_cv = 2 # 5
n_repeats_cv = 2 # 3

# Train parameters
num_boost_round_train = 2 # 60
num_mcc_points = 2 # 400

# directory of generated files
workdir = 'hpop_test_1'

# XGBoost searching parameters
param_grid = {'max_depth': [2], #[13, 14, 15], 
              'eta': [0.025, 0.03, 0.035],
              'silent': [1],
              'objective': ['binary:logistic'],
              'nthread': [16],
              'lambda': [3.5, 4, 4.5],
              'alpha': [0, 0.25], 
              'subsample': [0.85, 0.9, 0.95],
              'min_child_weight': [4.5, 5, 5.5],
              'booster': ['gbtree', 'dart'],
              'base_score': [0.0058], 
              'colsample_bytree': [0.5, 0.55, 0.6, 0.65]}

param_list = list(ParameterSampler(param_grid, 
    n_iter=100, random_state=np.random.randint(10**6)))

#%% 
# make work directory
rootdir = os.getcwd()
if not os.path.exists(workdir):
    os.mkdir(workdir)

# copy sample submission
if not os.path.exists(os.path.join(workdir, 'sample_submission.csv.zip')):
    copyfile('sample_submission.csv.zip', os.path.join(workdir, 'sample_submission.csv.zip'))

# copy hdf
if not os.path.exists(os.path.join(workdir, 'numeric_b1_b7_nf149.hdf')):
    copyfile('numeric_b1_b7_nf149.hdf', os.path.join(workdir, 'numeric_b1_b7_nf149.hdf'))

# create .py files
for param_id, param in enumerate(param_list):
    replace_lines = {
        23: "sys.path.insert(0, 'bosch_helper')\n", # should be changed when move to cluster
        28: "param_id = {}\n".format(param_id),
        29: "random_state = {}\n".format(np.random.randint(10**6)),
        30: "param = {}\n".format(param.__repr__()),
        35: "x = x.iloc[:, :30]\n",  # only in testing
        43: "    num_boost_round={},\n".format(num_boost_round_cv),
        44: "    n_splits={},\n".format(n_splits_cv),
        45: "    n_repeats={},\n".format(n_repeats_cv),
        54: "    num_boost_round={},\n".format(num_boost_round_train),
        59: "thresholds = np.linspace(0.01, 0.99, {})\n".format(num_mcc_points),       
        } 
    src = os.path.join(rootdir, 'hpopt_template.py')
    dst = os.path.join(rootdir, workdir, 'hpopt_test_{}.py'.format(param_id))
    change_template(src, dst, replace_lines)
    eol_win2unix(dst)

# copy bosch_helper
copyfile('../../bosch_helper/bosch_helper.py', os.path.join(workdir, 'bosch_helper.py'))