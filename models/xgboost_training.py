## Code for XGBoost training model 

# Packages 
import joblib
import pandas as pd
import xgboost as xgb
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint

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

## Basic Model - no tuning 
xgb_clf_basic = xgb.XGBClassifier()

# Parameters no tuning
print(xgb_clf_basic.get_params())

## Original Data 
xgb_clf_basic.fit(X_train, y_train)
y_test_pred_xgb = xgb_clf_basic.predict(X_test)
print(classification_report(y_test, y_test_pred_xgb)) # high accuracy but very unbalanced results

## Undersampling Method
xgb_clf_basic.fit(X_train_under, y_train_under)
y_test_pred_u_xgb = xgb_clf_basic.predict(X_test)
print(classification_report(y_test, y_test_pred_u_xgb)) # lower accuracy but more balanced results 

## Oversampling Method 
xgb_clf_basic.fit(X_train_over, y_train_over) 
y_test_pred_o_xgb = xgb_clf_basic.predict(X_test)
print(classification_report(y_test, y_test_pred_o_xgb)) # little better results to undersampling data 

# Hyperparameter tuning 
print(xgb_clf_basic.get_params()) # default parameters 

# Step 1: use random grid search 
n_estimators = randint(10, 500)
max_depth = randint(2, 15)
learning_rate = uniform(0.01, 0.5)
subsample = uniform(0.6, 0.4)
colsample_bytree = uniform(0.6, 0.4)
gamma = uniform(0, 0.5)

random_grid_xgb = {
    'n_estimators': n_estimators,
    'max_depth': max_depth, 
    'learning_rate': learning_rate, 
    'subsample': subsample, 
    'colsample_bytree': colsample_bytree,
    'gamma': gamma 
    }

xgb_clf_random = xgb.XGBClassifier()
random_search_xgb = RandomizedSearchCV(estimator = xgb_clf_random, 
                                       param_distributions = random_grid_xgb, 
                                       n_iter = 200, 
                                       scoring = 'accuracy',
                                       cv = 5, 
                                       verbose = 2, 
                                       n_jobs = -1)

random_search_xgb.fit(X_train_over, y_train_over)
bp_rs_xgb = random_search_xgb.best_params_
print(bp_rs_xgb)

# Try a model with best parameters from random search 
xgb_best_fit_rs = xgb.XGBClassifier(**bp_rs_xgb)
xgb_best_fit_rs.fit(X_train_over, y_train_over)

y_test_pred_xgb_rs = xgb_best_fit_rs.predict(X_test)
print(classification_report(y_test, y_test_pred_xgb_rs)) 


# Step 2: Grid Search 
param_grid_xgb = {
    'n_estimators': [200, 400, 600],
    'max_depth': [8, 13, 18], 
    'learning_rate': [0.1, 0.3, 0.5], 
    'subsample': [0.4, 0.6, 0.9], 
    'colsample_bytree': [0.5, 0.7, 0.9],
    'gamma': [0.008, 0.01, 0.1]
    }

print(param_grid_xgb)

xgb_clf_grid = xgb.XGBClassifier()
grid_search_xgb = GridSearchCV(estimator = xgb_clf_grid, 
                               param_grid = param_grid_xgb, 
                               scoring = 'accuracy',
                               cv = 5, 
                               n_jobs = -1, 
                               verbose = 2)

grid_search_xgb.fit(X_train_over, y_train_over)
bp_gs_xgb = grid_search_xgb.best_params_
print(bp_gs_xgb)

xgb_best_fit_gs = xgb.XGBClassifier(**bp_gs_xgb)
xgb_best_fit_gs.fit(X_train_over, y_train_over)
print(xgb_best_fit_gs.get_params)

# Save Model 
joblib.dump(xgb_best_fit_gs, 'xgb_best_model.pkl')

