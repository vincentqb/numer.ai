#!/usr/bin/env python

import pandas as pd
from time import clock

### Load data

train_file = 'dataset/numerai_training_data.csv'
test_file = 'dataset/numerai_tournament_data.csv'
output_file = 'predict.csv'

start = clock()
train = pd.read_csv( train_file )
test = pd.read_csv( test_file )
print('Loaded {:d} train and {:d} entries in {:.0f} seconds.'.format(len(train), len(test), clock() - start))

# Validation set is more reprensentative of tournament data
# train.drop( 'validation', axis = 1 , inplace = True )
train = train.drop(train[train.validation == 1].index)

### One-hot encode of categorical variable

# Check train and test have the same categories
assert( set( train.c1.unique()) == set( test.c1.unique()))

# Encode train column, then drop original column
train_dummies = pd.get_dummies( train.c1 )
train_num = pd.concat(( train.drop( 'c1', axis = 1 ), train_dummies.astype( int )), axis = 1 )

# Encode test column, then drop original column
test_dummies = pd.get_dummies( test.c1 )
test_num = pd.concat(( test.drop( 'c1', axis = 1 ), test_dummies.astype(int) ), axis = 1 )

### Select classifier

from sklearn.ensemble import RandomForestClassifier as rf
n_trees = 1000
clf = rf( n_estimators = n_trees, verbose = True )

### Cross validation

from sklearn.cross_validation import cross_val_score

start = clock()
scores = cross_val_score(clf, train, label)
print("Performed {:d}-fold cross validation in {:.0f} seconds with accuracy {:0.4f} +/- {:0.4f}.".format(
len(scores), clock() - start, scores.mean(), scores.std()))

### Fit training data

start = clock()
clf.fit( train_num.drop( 'target', axis = 1 ), train_num.target )
print("Fitted training data in {:.0f} seconds.".format(clock() - start))

### Extrapolate to test data

start = clock()
predict = clf.predict_proba( test_num.drop( 't_id', axis = 1 ))
print("Extrapolated to test data in {:.0f} seconds.".format(clock() - start))

### Save results

test_num['probability'] = predict[:,1]
test_num.to_csv( output_file, columns = ( 't_id', 'probability' ), index = None )
