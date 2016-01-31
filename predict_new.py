#!/usr/bin/env python

import pandas as pd
from time import clock

### Load data

train_file = 'dataset/numerai_training_data.csv'
test_file = 'dataset/numerai_tournament_data.csv'
predict_file = 'predict.csv'

start = clock()
train_data = pd.read_csv(train_file)
test_data = pd.read_csv(test_file)
print('Loaded {:d} train and {:d} test entries in {:.0f} seconds.'.format( 
    len(train_data), len(test_data), clock() - start))

# No need for validation flag for final training and extrapolation
train_data.drop('validation', axis = 1 , inplace = True)

# Separate data and target label
train_target = train_data['target']
train_data.drop('target', axis = 1, inplace = True)

### One-hot encode of categorical variable

# Check train and test have the same categories
assert(set(train_data['c1'].unique()) == set(test_data['c1'].unique()))

# Encode column in train, then drop original column
train_dummies = pd.get_dummies(train_data['c1'])
train_data = pd.concat((train_data.drop('c1', axis = 1), train_dummies.astype(int)), axis = 1)

# Encode column in test, then drop original column
test_dummies = pd.get_dummies(test_data['c1'])
test_data = pd.concat((test_data.drop('c1', axis = 1), test_dummies.astype(int)), axis = 1)

### Select classifier

from sklearn.ensemble import RandomForestClassifier as RF
clf = RF(n_estimators = 1000, verbose = True)

### Fit training data

start = clock()
clf.fit(train_data, train_target)
print("Fitted in {:.0f} seconds.".format(clock() - start))

### Extrapolate

start = clock()
predict = clf.predict_proba(test_data)
print("Extrapolated in {:.0f} seconds.".format(clock() - start))

### Save results

test_data['probability'] = predict[:,1]
test_data.to_csv(predict_file, columns = ('t_id', 'probability'), index = None)
