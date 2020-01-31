# Code for Kaggle Titanic dataset
# tabular, binary classification

# Mount Google Drive 
from google.colab import drive
drive.mount('/content/gdrive', force_remount=True)

root_path = '/content/gdrive/My Drive/google_colab/'
import os
os.chdir(root_path)

# Import data from kaggle API
!mkdir ~/.kaggle
!cp kaggle.json ~/.kaggle/
!kaggle competitions download -c titanic --force

# libraries
import os
import sys
import pdb
import itertools

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.utils import class_weight
from sklearn.model_selection import RandomizedSearchCV, KFold
from scipy.stats import randint as sp_randint 
from scipy.stats import uniform as sp_unif
import xgboost as xgb

# load data
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
all_df = pd.concat([train_df, test_df]) 

# handle NAs
all_df.isna().sum() / all_df.shape[0] # check NA percentage for each column
all_df = all_df.drop(['Cabin'], axis = 1) # drop cabin since it has 77% missing 
all_df.Age = all_df.Age.fillna(value = np.mean(all_df.Age))
all_df.Embarked = all_df.Embarked.fillna(method = 'ffill')

# create a new title feature
all_df['Title'] = all_df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
all_df['Title'] = all_df['Title'].replace(['Lady', 'Countess','Capt', 'Col',\
 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
all_df['Title'] = all_df['Title'].replace('Mlle', 'Miss')
all_df['Title'] = all_df['Title'].replace('Ms', 'Miss')
all_df['Title'] = all_df['Title'].replace('Mme', 'Mrs')
#all_df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean() # check distribution

# create a new age group feature
def find_age_group(age):
  if age > 0 and age < 20:
    age_group = 1
  elif age >= 20 and age < 40:
    age_group = 2
  elif age >= 40 and age < 60:
    age_group = 3
  elif age >= 60 and age < 80:
    age_group = 4
  else:
    age_group = 5
  return age_group
all_df['Age_group'] = all_df.Age.apply(find_age_group)

# drop some columns
all_df = all_df.drop(['Name', 'Ticket'], axis = 1)

# label encoding of non-numeric columns
nonnumeric_columns = ['Sex', 'Embarked', 'Title']
le = LabelEncoder()
for feature in nonnumeric_columns:
    all_df[feature] = le.fit_transform(all_df[feature])

# split train and test
X_train = all_df[0:train_df.shape[0]]
X_train = X_train.drop(['Survived'], axis = 1)
y_train = train_df.Survived

X_test = all_df[train_df.shape[0]::]
X_test = X_test.drop(['Survived'], axis = 1)

# compute class weight
cw = class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)

# find the best combo of parameters using random search and kfold 
# specify its possible parameters distributions
gbm  = xgb.XGBClassifier()
param_dist = {
    'learning_rate': sp_unif(1e-6,0.15),
    'n_estimators': sp_randint(50,300),
    'max_depth': sp_randint(2,8),
    'gamma': sp_unif(0,1),
    'subsample': sp_unif(0,1),
    'colsample_bytree': sp_unif(0,1) 
}

random_search = RandomizedSearchCV(
    gbm, 
    param_distributions = param_dist,
    cv = KFold(3),
    n_iter = 100, # number of random draws
    random_state = 0
)

random_search.fit(X_train, y_train)
random_search.best_params_

# train xgb classifier with the best params
gbm = xgb.XGBClassifier(
    seed             = 0,
    silent           = 1,
    learning_rate    = random_search.best_params_["learning_rate"], 
    n_estimators     = random_search.best_params_["n_estimators"],
    max_depth        = random_search.best_params_["max_depth"],
    gamma            = random_search.best_params_["gamma"],
    subsample        = random_search.best_params_["subsample"],
    colsample_bytree = random_search.best_params_["colsample_bytree"],
    class_weight     = cw
).fit(X_train, y_train)

train_pred = gbm.predict(X_train)
test_pred = gbm.predict(X_test)
accuracy_score(y_train, train_pred) # around 91% accuracy on the training set

# prepare for submission
submission = pd.DataFrame({ 'PassengerId': test_df['PassengerId'],
                            'Survived': test_pred})
submission.to_csv("submission.csv", index=False)