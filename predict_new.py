#!/usr/bin/env python

import pandas as pd
from time import clock

### Load data

train_file = 'dataset/numerai_training_data.csv'
test_file = 'dataset/numerai_tournament_data.csv'
predict_file = 'predict.csv'

start = clock()
train = pd.read_csv(train_file)
test = pd.read_csv(test_file)
print('Loaded {:d} train and {:d} test entries in {:.0f} seconds.'.format( 
    len(train), len(test), clock() - start))

# No need for validation flag for final training and extrapolation
train.drop('validation', axis = 1 , inplace = True)

### One-hot encode of categorical variable

# Check train and test have the same categories
assert(set(train['c1'].unique()) == set(test['c1'].unique()))

# Encode train column, then drop original column
train_dummies = pd.get_dummies(train['c1'])
train = pd.concat((train.drop('c1', axis = 1), train_dummies.astype(int)), axis = 1)

# Encode test column, then drop original column
test_dummies = pd.get_dummies(test['c1'])
test = pd.concat((test.drop('c1', axis = 1), test_dummies.astype(int)), axis = 1)

### Select classifier

from sklearn.ensemble import RandomForestClassifier as rf
clf = rf(n_estimators = 1000, verbose = True)

### Fit training data

start = clock()
clf.fit(train.drop('target', axis = 1), train.target)
print("Fitted in {:.0f} seconds.".format(clock() - start))

### Extrapolate

start = clock()
predict = clf.predict_proba(test.drop('t_id', axis = 1))
print("Extrapolated in {:.0f} seconds.".format(clock() - start))

### Save results

test['probability'] = predict[:,1]
test.to_csv(predict_file, columns = ('t_id', 'probability'), index = None)
