#!/usr/bin/env python

"Load data, create the validation split, train a random forest, evaluate"
"uncomment the appropriate lines to save processed data to disk"

import pandas as pd
from time import clock

from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.metrics import roc_auc_score as AUC
from sklearn.metrics import accuracy_score as accuracy

### Load data

train_file = 'dataset/numerai_training_data.csv'

start = clock()
data = pd.read_csv(train_file)
print('Loaded {:d} entries in {:.0f} seconds.'.format( 
    len(train), clock() - start))

# Validation set is more reprensentative of tournament data

# Split train into test and train
iv = data.validation == 1
test_data = data[iv].copy()
train_data = data[~iv].copy()

# Separate data and label
train_label = train_data['target']
train_data.drop('target', axis = 1, inplace = True)
test_label = test_data['target']
test_data.drop('target', axis = 1, inplace = True)

# Validation flag no longer needed
train.drop( 'validation', axis = 1 , inplace = True )

### One-hot encode of categorical variable

# One-hot encode of categorical variable
# Encode column in train, then drop original column
train_dummies = pd.get_dummies(train_data['c1'])
train = pd.concat((train_data.drop('c1', axis = 1), train_dummies.astype(int)), axis = 1)

# train, predict, evaluate

n_trees = 100

rf = RF( n_estimators = n_trees, verbose = True )
rf.fit( train_num.drop( 'target', axis = 1 ), train_num.target )

p = rf.predict_proba( val_num.drop( 'target', axis = 1 ))
p_bin = rf.predict( val_num.drop( 'target', axis = 1 ))

acc = accuracy( val_num.target.values, p_bin )
auc = AUC( val_num.target.values, p[:,1] )
print "AUC: {:.2%}, accuracy: {:.2%}".format( auc, acc )
	
# AUC: 51.40%, accuracy: 51.14%	/ 100 trees
# AUC: 52.16%, accuracy: 51.62%	/ 1000 trees
