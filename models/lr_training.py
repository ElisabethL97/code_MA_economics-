## Code for Logistic Regression training model 

# Packages 
import joblib
import pandas as pd
import numpy as np
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV


# Load preprocessed data from CSV
file_path = 'preprocessing/'
X = pd.read_csv(file_path + 'X_preprocessed.csv')
y = pd.read_csv(file_path + 'y_preprocessed.csv')
y = y.values.ravel()


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Oversampling
oversampler = RandomOverSampler(sampling_strategy='auto', random_state=42)
X_train_over, y_train_over = oversampler.fit_resample(X_train, y_train)

# Undersampling
undersampler = RandomUnderSampler(sampling_strategy='auto', random_state=42)
X_train_under, y_train_under = undersampler.fit_resample(X_train, y_train)

print(X_train_over.shape) # over 1 mil rows
print(X_train_under.shape) # 94k rows 

## Basic Model - no tuning 
lr_clf_basic = LogisticRegression()
print(lr_clf_basic.get_params())

## Original data 
lr_clf_basic.fit(X_train, y_train) 
y_test_pred_lr = lr_clf_basic.predict(X_test)
print(classification_report(y_test, y_test_pred_lr, zero_division=1)) # high accuracy but very unbalanced results


## Undersampling Method
lr_clf_basic.fit(X_train_under, y_train_under) 
y_test_pred_u_lr = lr_clf_basic.predict(X_test)
print(classification_report(y_test, y_test_pred_u_lr)) # lower accuracy but more balanced results 

## Oversampling Method 
lr_clf_basic.fit(X_train_over, y_train_over) 
y_test_pred_o_lr = lr_clf_basic.predict(X_test)
print(classification_report(y_test, y_test_pred_o_lr)) # very similar results to undersampling data 

# Hyperparameter tuning
print(lr_clf_basic.get_params()) # default parameters 

# Grid Search 
param_grid_lr = [
    {
        'penalty': ['l1', 'l2'],
        'C': np.logspace(-4, 4, 20),
        'solver': ['liblinear'],
        'max_iter': [100, 500, 1000, 1500],
        'class_weight': [None, 'balanced']
    },
    {
        'penalty': ['elasticnet'],
        'C': np.logspace(-4, 4, 20),
        'solver': ['saga'],
        'l1_ratio': [0.2, 0.5, 0.8],
        'max_iter': [100, 500, 1000, 1500],
        'class_weight': [None, 'balanced']
    },
    {
        'penalty': ['none'],
        'solver': ['newton-cg', 'lbfgs', 'sag'],
        'max_iter': [100, 500, 1000, 1500],
        'class_weight': [None, 'balanced']
    }
]


lr_clf_grid = LogisticRegression()
grid_search_lr = GridSearchCV(estimator = lr_clf_grid, 
                               param_grid = param_grid_lr, 
                               cv = 5, 
                               n_jobs = -1, 
                               verbose = 2)

grid_search_lr.fit(X_train_under, y_train_under)
bp_gs_lr = grid_search_lr.best_params_
print(bp_gs_lr)

lr_best_fit_gs = LogisticRegression(**bp_gs_lr)
lr_best_fit_gs.fit(X_train_under, y_train_under)
print(lr_best_fit_gs.get_params)

# Save Model 
joblib.dump(lr_best_fit_gs, 'lr_best_model.pkl')


