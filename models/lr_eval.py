# Packages 
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_curve, auc

# Load preprocessed data from CSV
file_path = 'preprocessing/'
X = pd.read_csv(file_path + 'X_preprocessed.csv')
y = pd.read_csv(file_path + 'y_preprocessed.csv')
y = y.values.ravel()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Load model
file_path = 'saved_models/'
lr_best_fit_gs = joblib.load(file_path + 'lr_best_model.pkl')

# optimal parameters
print(lr_best_fit_gs.get_params())

# Show model performance 
y_test_pred_lr_gs = lr_best_fit_gs.predict(X_test)
print(classification_report(y_test, y_test_pred_lr_gs)) 

# Confusion Matrix
cm_lr = confusion_matrix(y_test, y_test_pred_lr_gs)

# Normalize the confusion matrix
cm_lr_norm = cm_lr.astype('float') / cm_lr.sum(axis=1)[:, np.newaxis]

# Create a custom colormap
cmap_custom = sns.diverging_palette(10, 150, as_cmap=True)

# Plot the confusion matrix 
plt.figure(figsize=(8, 6))
sns.heatmap(cm_lr_norm, annot=True, cmap=cmap_custom, annot_kws={"size": 16},
            vmin=0, vmax=1)  
plt.title('Confusion Matrix - Logistic Regression')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()


# AUC & ROC curve
y_val_roc_lr = lr_best_fit_gs.predict_proba(X_test)[:, 1]

fpr_lr, tpr_lr, thresholds_lr= roc_curve(y_test, y_val_roc_lr)
roc_auc_lr = auc(fpr_lr, tpr_lr)

# Plot ROC curve 
plt.figure(figsize=(8, 6))
plt.plot(fpr_lr, tpr_lr, color='darkorange', lw=2, label='ROC curve (AUC = {:.2f})'.format(roc_auc_lr))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('Specificity - FP rate')
plt.ylabel('Sensitivity - TP rate')
plt.title('ROC Curve - Logistic Regression')
plt.legend(loc='lower right')
plt.show()


# Feature Importance 
coef_lr = lr_best_fit_gs.coef_[0]
feature_names = X_train.columns

# Mapping coefficients to feature names
fi_df_lr = pd.DataFrame({'Feature': feature_names, 'Coefficient': coef_lr})
fi_df_lr['AbsoluteCoefficient'] = np.abs(fi_df_lr['Coefficient'])
fi_df_lr = fi_df_lr.sort_values(by='AbsoluteCoefficient', ascending=True)

palette = sns.color_palette("Blues", n_colors=len(fi_df_lr))

# Plotting the feature importance
plt.figure(figsize=(10, 6))
plt.barh(fi_df_lr['Feature'], fi_df_lr['AbsoluteCoefficient'], color = palette)
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importance - Logistic Regression')
plt.show()


