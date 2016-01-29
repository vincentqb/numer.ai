#!/usr/bin/env python

"cross-validation"

import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer, PolynomialFeatures
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, RobustScaler, StandardScaler
from sklearn.linear_model import LogisticRegression as LR
from sklearn.metrics import roc_auc_score as AUC
from sklearn.metrics import accuracy_score as accuracy

train_file = 'dataset/numerai_training_data.csv'

start = clock()
train_frame = pd.read_csv(train_file)
print('Loaded {:d} train entries in {:.0f} seconds.'.format( 
    len(train), clock() - start))

# Remove validation column, not used here
train_frame.drop('validation', axis = 1 , inplace = True)

# Separate train data and label
label = train['target']
train.drop('target', axis = 1, inplace = True)

### One-hot encode of categorical variable

# Encode column in train, then drop original column
train_dummies = pd.get_dummies(train['c1'])
train = pd.concat((train_frame.drop('c1', axis = 1), train_dummies.astype(int)), axis = 1)

### Cross validation

from sklearn.cross_validation import cross_val_score

start = clock()
scores = cross_val_score(clf, train, label.target, scoring = 'roc_auc', cv = 10, verbose = 1)
print("Performed {:d}-fold cross validation in {:.0f} seconds with ROC AUC {:0.4f} +/- {:0.4f}.".format(
    len(scores), clock() - start, scores.mean(), scores.std() ))
