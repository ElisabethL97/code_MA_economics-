## Code for Random Forest training model 

# Packages
import joblib
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
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

print(X_train_over.shape) # over 1 mil rows
print(X_train_under.shape) # 94k rows 

## Basic Model - no tuning 
rf_clf_basic = RandomForestClassifier()

# Parameters no tuning
print(rf_clf_basic.get_params())

## Original data 
rf_clf_basic.fit(X_train, y_train) 
y_test_pred_rf = rf_clf_basic.predict(X_test)
print(classification_report(y_test, y_test_pred_rf)) # high accuracy but very unbalanced results

## Undersampling Method
rf_clf_basic.fit(X_train_under, y_train_under) 
y_test_pred_u_rf = rf_clf_basic.predict(X_test)
print(classification_report(y_test, y_test_pred_u_rf)) # lower accuracy but more balanced results 

## Oversampling Method 
rf_clf_basic.fit(X_train_over, y_train_over) 
y_test_pred_o_rf = rf_clf_basic.predict(X_test)
print(classification_report(y_test, y_test_pred_o_rf))

# Hyperparameter tuning 
print(rf_clf_basic.get_params()) # default parameters 

# Step 1: use random grid search 
n_estimators = randint(10, 500)
max_features = ['auto', 'sqrt']
max_depth = randint(2, 100)
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]
boostrap = [True, False]

random_grid_rf = {'n_estimators': n_estimators, 
                  'max_features': max_features, 
                  'max_depth': max_depth, 
                  'min_samples_split': min_samples_split, 
                  'min_samples_leaf': min_samples_leaf, 
                  'bootstrap': boostrap}

rf_clf_random = RandomForestClassifier()
random_search_rf = RandomizedSearchCV(estimator = rf_clf_random, 
                                      param_distributions = random_grid_rf, 
                                      n_iter = 200, 
                                      cv = 5, 
                                      verbose = 2, 
                                      n_jobs = -1)

random_search_rf.fit(X_train_under, y_train_under)
bp_rs_rf = random_search_rf.best_params_
print(bp_rs_rf)

# Try a model with best parameters from random search 
rf_best_fit_rs = RandomForestClassifier(**bp_rs_rf)
rf_best_fit_rs.fit(X_train_under, y_train_under)

y_test_pred_rf_rs = rf_best_fit_rs.predict(X_test)
print(classification_report(y_test, y_test_pred_rf_rs)) 

# Step 2: Grid Search 
param_grid_rf = {'bootstrap': [True], 
                 'max_depth': [7, 9, 15], 
                 'max_features': ['sqrt'], 
                 'min_samples_leaf': [1, 3, 5], 
                 'min_samples_split': [8, 10, 13], 
                 'n_estimators': [20, 45, 80]}

rf_clf_grid = RandomForestClassifier()
grid_search_rf = GridSearchCV(estimator = rf_clf_grid, 
                              param_grid = param_grid_rf, 
                              cv = 5, 
                              n_jobs = -1, 
                              verbose = 2)

grid_search_rf.fit(X_train_under, y_train_under)
bp_gs_rf = grid_search_rf.best_params_
print(bp_gs_rf)

rf_best_fit_gs = RandomForestClassifier(**bp_gs_rf)
rf_best_fit_gs.fit(X_train_under, y_train_under)
print(rf_best_fit_gs.get_params)

# Save Model 
joblib.dump(rf_best_fit_gs, 'rf_best_model.pkl')


