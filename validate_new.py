#!/usr/bin/env python

"Load data, create the validation split, train a random forest, evaluate"
"uncomment the appropriate lines to save processed data to disk"

import pandas as pd
from time import clock

from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.metrics import roc_auc_score as AUC
from sklearn.metrics import accuracy_score as accuracy

### Load data

input_file = 'dataset/numerai_training_data.csv'

start = clock()
data = pd.read_csv(input_file)
print('Loaded {:d} entries in {:.0f} seconds.'.format( 
    len(data), clock() - start))

### Validation set is more reprensentative of tournament data

# Identify validation set
iv = data.validation == 1

# Validation flag no longer needed
data.drop( 'validation', axis = 1 , inplace = True )

# Split train into test and train
test_data = data[iv].copy()
train_data = data[~iv].copy()

# Separate data and target label
train_target = train_data['target']
train_data.drop('target', axis = 1, inplace = True)
test_target = test_data['target']
test_data.drop('target', axis = 1, inplace = True)

### One-hot encode of categorical variable

# Encode column in train, then drop original column
train_dummies = pd.get_dummies(train_data['c1'])
train_data = pd.concat((train_data.drop('c1', axis = 1), train_dummies.astype(int)), axis = 1)

# Encode column in train, then drop original column
test_dummies = pd.get_dummies(test_data['c1'])
test_data = pd.concat((test_data.drop('c1', axis = 1), test_dummies.astype(int)), axis = 1)

### Select classifier

from sklearn.ensemble import RandomForestClassifier as RF
rf = RF(n_estimators = 10, verbose = True)

### Fit and extrapolate

start = clock()
rf.fit(train_data, train_target)
print("Fitted in {:.0f} seconds.".format(clock() - start))

start = clock()
predict = rf.predict_proba(test_data)
predict_bin = rf.predict(test_data)
print("Extrapolated in {:.0f} seconds.".format(clock() - start))

### Compute ROC AUC and accuracy

acc = accuracy(test_target.values, predict_bin)
auc = AUC(test_target.values, predict[:,1])
print "AUC: {:.2%}. Accuracy: {:.2%}.".format(auc, acc)

"""
Results

RF(n_estimators = 10, verbose = True)
Fitted in 3 seconds. Extrapolated in 0 seconds.
AUC: 50.67%. Accuracy: 49.67%.
"""
