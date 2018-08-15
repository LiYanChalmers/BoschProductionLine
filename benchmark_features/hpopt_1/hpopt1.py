# -*- coding: utf-8 -*-
"""
Prepare data
numeric.to_hdf('numeric_b1_b7_nf149.hdf', 'numeric')
"""

import sys
sys.path.insert(0, '../../bosch_helper')
from bosch_helper import *

param_grid = {'max_depth': [13, 14, 15, 16], 
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

#%% Load data of both train and test sets
# load numeric data
numeric = pd.read_hdf('../../data/data.hdf', 'numeric')

# labels for the train set
y_train = numeric.loc[('train', slice(None)), 'Response']
y_train = y_train.astype(np.int8)

# Import names of important features
important_features = pd.read_csv(
    '../benchmark_1/important_numeric_features.csv', 
    index_col=0, header=None)
important_features = list(important_features.values.ravel())

numeric = numeric[important_features]
numeric.index.names = ['set', 'Id']

#%% Load features: station flow
date_train = pd.read_csv(
    '../benchmark_2/train_station_flow.csv.gz', 
    index_col=0, header=None)
date_test = pd.read_csv(
    '../benchmark_2/test_station_flow.csv.gz', 
    index_col=0, header=None)

# Change index and column names
station_flow = pd.concat((date_train, date_test), keys=['train', 'test'])
station_flow.index.names = ['set', 'Id']
station_flow.columns = ['hash_station_flow0']

# Encode hash 
le = LabelEncoder()
station_flow['hash_station_flow0'] = le.fit_transform(station_flow)

# Join to numeric
numeric = numeric.join(station_flow)

del station_flow
gc.collect()

#%% Load features: benchmark 3, consective Id chunk
start_chunk = pd.read_csv('../benchmark_3/start_chunk.csv.gz', index_col=0)

# Group start chunks by train and test sets
start_chunk_train = start_chunk.loc[start_chunk.Response!=-1].drop(
    ['Response'], axis=1)
start_chunk_test = start_chunk.loc[start_chunk.Response==-1].drop(
    ['Response'], axis=1)
start_chunk = pd.concat((start_chunk_train, start_chunk_test), 
    keys=['train', 'test'])

start_chunk.index.names = ['set', 'Id']

# Join to numeric
numeric = numeric.join(start_chunk)

del start_chunk, start_chunk_test, start_chunk_train
gc.collect()

#%% Load features: benchmark 4, neighor time and response records
n = pd.read_csv('../benchmark_4/benchmark_4_neighbors.csv.gz', index_col=0)

# Group by train and test 
neighbor_train = n.loc[n.Response!=-1]
neighbor_train.drop(['Response'], axis=1, inplace=True)

neighbor_test = n.loc[n.Response==-1]
neighbor_test.drop(['Response'], axis=1, inplace=True)

neighbor = pd.concat((neighbor_train, neighbor_test), keys=['train', 'test'])

neighbor.index.names = ['set', 'Id']

# Join to numeric
numeric = numeric.join(neighbor)

del neighbor, neighbor_test, neighbor_train, n
gc.collect()

#%% Load features: benchmark 6, neighbor numeric features
numeric.sort_index(by=['Id'], inplace=True)
numeric = numeric.join(numeric[important_features].shift(), 
    rsuffix='_previous')
numeric = numeric.join(numeric[important_features].shift(-1),
    rsuffix='_next')

#%% Load features: benchmark 7, time features without MeanTimeDiff
time_features = pd.read_hdf('../benchmark_7/time_features_diff.hdf', 
    'time_features')
time_features.drop(['time_start', 'time_end', 'time_duration', 'Response'], 
    axis=1, inplace=True)
time_features.drop(time_features.columns[-40:], axis=1, inplace=True)

time_features.index.names = ['set', 'Id']

# Join to numeric
numeric = numeric.join(time_features)

del time_features
gc.collect()

#%% Save numeric data to a HDF for later use
for c in tqdm.tqdm(numeric.columns):
    if numeric[c].dtype==np.float64:
        numeric[c] = numeric[c].astype(np.float16)

numeric.to_hdf('numeric_b1_b7_nf149.hdf', 'numeric')
y_train.to_hdf('numeric_b1_b7_nf149.hdf', 'y_train')