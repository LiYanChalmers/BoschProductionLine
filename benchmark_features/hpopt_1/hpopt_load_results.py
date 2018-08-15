# -*- coding: utf-8 -*-
"""
Load results from hpopt_template.py
"""

import sys
sys.path.insert(0, '../../bosch_helper')
from bosch_helper import *

#%%
param_id = 0

results = read_pickle('results_hpopt_{}.pickle'.format(param_id))
sub = pd.read_csv("submission_hpopt_{}.csv.gz".format(param_id), index_col=0)

for k in results.keys():
    print(k)
    
