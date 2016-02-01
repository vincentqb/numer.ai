#!/usr/bin/env python

"""
Cross-validation with a few transformers.
"""

import pandas as pd
from time import clock

### Load and prepare data

train_file = 'dataset/numerai_training_data.csv'

start = clock()
train_frame = pd.read_csv(train_file)
print('Loaded {:d} train entries in {:.0f} seconds.'.format( 
    len(train_frame), clock() - start))

# Remove validation column, not used here
train_frame.drop('validation', axis = 1 , inplace = True)

# Separate train data and label
label = train_frame['target']
train_frame.drop('target', axis = 1, inplace = True)

# One-hot encode of categorical variable
# Encode column in train, then drop original column
train_dummies = pd.get_dummies(train_frame['c1'])
train = pd.concat((train_frame.drop('c1', axis = 1), train_dummies.astype(int)), axis = 1)

### Select transformer

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer, PolynomialFeatures, MaxAbsScaler, MinMaxScaler, RobustScaler, StandardScaler

poly_scaled = Pipeline([ ('poly', PolynomialFeatures()), ( 'scaler', MinMaxScaler()) ])

transformers = [ MaxAbsScaler(), MinMaxScaler(), RobustScaler(), StandardScaler(), 
                 Normalizer( norm = 'l1' ), Normalizer( norm = 'l2' ), Normalizer( norm = 'max' ), 
                 PolynomialFeatures(), poly_scaled ]

### Select classifier

from sklearn.linear_model import LogisticRegression as LR
clf = LR()

### Cross validation

from sklearn.cross_validation import cross_val_score

for transfomer in transfomers:

    print transformer
    start = clock()
    train_transformed = transformer.fit_transform(train, axis = 1)
    scores = cross_val_score(clf, train_transformed, label, scoring = 'roc_auc', cv = 10, verbose = 1)
    print(
        "Performed {:d}-fold cross validation in {:.0f} seconds with ROC AUC: mean {:0.4f} std {:0.4f}.".format(
        len(scores), clock() - start, scores.mean(), scores.std() ))

"""
Results
"""
