## Code for Neural Network training model 

# Packages 
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import keras
from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf
import keras_tuner as kt


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

# Basic Model - no tuning
nn_clf_basic = Sequential()
nn_clf_basic.add(Dense(units=32, activation='relu', input_shape=(17,))) # amount of features x2
nn_clf_basic.add(Dense(units=64, activation='relu', input_shape=(17,))) # shape of first layer x2
nn_clf_basic.add(Dense(units=1, activation='sigmoid')) # output layer 

# Compile the model
nn_clf_basic.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

print(nn_clf_basic.summary())

## Original data 
nn_clf_basic.fit(X_train, y_train, epochs=10) 
y_test_pred_nn = nn_clf_basic.predict(X_test)
y_test_pred_nn_bin = (y_test_pred_nn > 0.5).astype(int)
print(classification_report(y_test, y_test_pred_nn_bin)) # high accuracy but very unbalanced results

## Undersampling Method
nn_clf_basic.fit(X_train_under, y_train_under, epochs=10) 
y_test_pred_u_nn = nn_clf_basic.predict(X_test)
y_test_pred_u_nn_bin = (y_test_pred_u_nn > 0.5).astype(int)
print(classification_report(y_test, y_test_pred_u_nn_bin)) # lower accuracy but more balanced results 

## Oversampling Method 
nn_clf_basic.fit(X_train_over, y_train_over, epochs=10) 
y_test_pred_o_nn = nn_clf_basic.predict(X_test)
y_test_pred_o_nn_bin = (y_test_pred_o_nn > 0.5).astype(int)
print(classification_report(y_test, y_test_pred_o_nn_bin))

# Hyperparameter tuning 

# Define a function to create your neural network model
def nn_model_tuned(hp):
    model = tf.keras.Sequential()

    hp_activation = hp.Choice('activation', values=['relu', 'tanh'])
    hp_layer_1 = hp.Int('layer_1', min_value=1, max_value=1000, step=100)
    hp_layer_2 = hp.Int('layer_2', min_value=1, max_value=1000, step=100)
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4, 1e-5])

    model.add(tf.keras.layers.Dense(units=hp_layer_1, activation=hp_activation, input_shape=(17,)))
    model.add(tf.keras.layers.Dense(units=hp_layer_2, activation=hp_activation))
    model.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

    model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=hp_learning_rate),
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=['accuracy'])

    return model


tuner = kt.Hyperband(nn_model_tuned, 
                     objective = ['accuracy'],
                     max_epochs = 10, 
                     factor = 3)

stop_early = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 5)

tuner.search(X_train_under, y_train_under, epochs = 50, validation_split = 0.2, callbacks = [stop_early])
best_params_nn = tuner.get_best_hyperparameters(num_trials=1)[0]

# Get best Hyperparameters
# Assuming you have a tuner object named 'tuner'
best_trial = tuner.oracle.get_best_trials(1)[0]  
best_hyperparameters_nn = best_trial.hyperparameters.values  
print(best_hyperparameters_nn)

# Fit best model 
nn_best_model = tuner.hypermodel.build(best_params_nn)
nn_best_model_hist = nn_best_model.fit(X_train_under, y_train_under, epochs = 50, 
                                       validation_split = 0.2, callbacks = [stop_early])

nn_best_model.save("nn_best_model.keras")
