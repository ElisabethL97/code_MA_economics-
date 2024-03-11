# Packages 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import load_model
from sklearn.metrics import roc_curve, auc
from tensorflow.keras.utils import plot_model
import pydot
import graphviz


# Load preprocessed data from CSV
file_path = 'preprocessing/'
X = pd.read_csv(file_path + 'X_preprocessed.csv')
y = pd.read_csv(file_path + 'y_preprocessed.csv')
y = y.values.ravel()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Load model
file_path = 'saved_models/'
nn_best_model = load_model(file_path + "nn_best_model.keras")

# optimal model
print(nn_best_model.summary())

# Illustrate NN Model 


# Show model performance 
y_test_pred_opt_nn = nn_best_model.predict(X_test)
y_test_pred_nn_opt_bin = (y_test_pred_opt_nn > 0.5).astype(int)
print(classification_report(y_test, y_test_pred_nn_opt_bin))

# Confusion Matrix
cm_nn = confusion_matrix(y_test, y_test_pred_nn_opt_bin)

# Normalize the confusion matrix
cm_nn_norm = cm_nn.astype('float') / cm_nn.sum(axis=1)[:, np.newaxis]

# Create a custom colormap
cmap_custom = sns.diverging_palette(10, 150, as_cmap=True)

# Plot the confusion matrix 
plt.figure(figsize=(8, 6))
sns.heatmap(cm_nn_norm, annot=True, cmap=cmap_custom, annot_kws={"size": 16},
            vmin=0, vmax=1)  
plt.title('Confusion Matrix - Neural Network')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# ROC curve
y_val_roc_nn = nn_best_model.predict(X_test)

# Calculate ROC curve and AUC for text set
fpr_nn, tpr_nn, thresholds_nn = roc_curve(y_test, y_val_roc_nn)
roc_auc_nn = auc(fpr_nn, tpr_nn)

# Plot ROC curve for validation set
plt.figure(figsize=(8, 6))
plt.plot(fpr_nn, tpr_nn, color='darkorange', lw=2, label='ROC curve (AUC = {:.2f})'.format(roc_auc_nn))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('Specificity - FP Rate')
plt.ylabel('Sensitivity - TP Rate')
plt.title('ROC Curve - Neural Network')
plt.legend(loc='lower right')
plt.show()




