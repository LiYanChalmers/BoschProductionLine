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
num_boost_round_cv = 80 # 80
n_splits_cv = 5 # 5
n_repeats_cv = 3 # 3

# Train parameters
num_boost_round_train = 60 # 60
num_mcc_points = 400 # 400

# directory of generated files
workdir = 'hpop_test_1'

# file names 
task_name = 'ht'
task_name_short = 'ht'

# cluster parameters
project_hpc = 'C3SE2018-1-15' # 'C3SE2018-1-15' or 'C3SE407-15-3'
cluster_hpc = 'hebbe' # 'hebbe', 'glenn'
node_hpc = 1
threads_hpc = 20
mem_hpc = 128
days_hpc = 0
hours_hpc = 8
minutes_hpc = 0
seconds_hpc = 0

# XGBoost searching parameters
n_params = 3
param_grid = {'max_depth': [13, 14, 15], #[13, 14, 15], 
              'eta': [0.025, 0.03, 0.035],
              'silent': [1],
              'objective': ['binary:logistic'],
              'nthread': [20],
              'lambda': [3.5, 4, 4.5],
              'alpha': [0, 0.25], 
              'subsample': [0.85, 0.9, 0.95],
              'min_child_weight': [4.5, 5, 5.5],
              'booster': ['gbtree', 'dart'],
              'base_score': [0.0058], 
              'colsample_bytree': [0.5, 0.55, 0.6, 0.65]}

# Create param list
param_list = list(ParameterSampler(param_grid, 
    n_iter=n_params, random_state=np.random.randint(10**6)))

#%% Python files
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
        23: "sys.path.insert(0, 'bosch_helper')\n",
        28: "param_id = {}\n".format(param_id),
        29: "random_state = {}\n".format(np.random.randint(10**6)),
        30: "param = {}\n".format(param.__repr__()),
        35: "\n", #"x = x.iloc[:, :30]\n",  # only in testing
        43: "    num_boost_round={},\n".format(num_boost_round_cv),
        44: "    n_splits={},\n".format(n_splits_cv),
        45: "    n_repeats={},\n".format(n_repeats_cv),
        54: "    num_boost_round={},\n".format(num_boost_round_train),
        59: "thresholds = np.linspace(0.01, 0.99, {})\n".format(num_mcc_points),
		74: "sub.to_csv('{}_{}.csv.gz', compression='gzip')\n".format(task_name, param_id),
		79: "save_pickle(results, '{}_{}.pickle')\n".format(task_name, param_id)
        } 
    src = os.path.join(rootdir, 'hpopt_template.py')
    dst = os.path.join(rootdir, workdir, '{}_{}.py'.format(task_name, param_id))
    change_template(src, dst, replace_lines)
    eol_win2unix(dst)

# copy bosch_helper
copyfile('../../bosch_helper/bosch_helper.py', os.path.join(workdir, 'bosch_helper.py'))

#%% bash files
# create .py files
for param_id, param in enumerate(param_list):
    replace_lines = {
        1: "#SBATCH -A {}\n".format(project_hpc), 
        2: "#SBATCH -p {}\n".format(cluster_hpc),
        3: "#SBATCH -J {}_{}\n".format(task_name_short, param_id),
        4: "#SBATCH -N {}\n".format(node_hpc),
        5: "#SBATCH -n {}\n".format(threads_hpc),  
        6: "#SBATCH -C MEM{}\n".format(mem_hpc),
        7: "#SBATCH -t {}-{}:{}:{}\n".format(days_hpc, hours_hpc, minutes_hpc, seconds_hpc),
        8: "#SBATCH -o {}_{}.stdout\n".format(task_name, param_id),
        9: "#SBATCH -e {}_{}.stderr\n".format(task_name, param_id),
        17: "pdcp {}_{}.py $TMPDIR\n".format(task_name, param_id),
        22: "python {}_{}.py\n".format(task_name, param_id)
        } 
    src = os.path.join(rootdir, 'hpopt_template.sh')
    dst = os.path.join(rootdir, workdir, '{}_{}.sh'.format(task_name, param_id))
    change_template(src, dst, replace_lines)
    eol_win2unix(dst)

#%% copy run_sbatch.py
copyfile('run_sbatch.py', os.path.join(workdir, 'run_sbatch.py'))