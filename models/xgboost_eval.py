## Code for XGBoost model evaluation

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
from scipy.stats import uniform, randint

# Load preprocessed data from CSV
file_path = 'preprocessing/'
X = pd.read_csv(file_path + 'X_preprocessed.csv')
y = pd.read_csv(file_path + 'y_preprocessed.csv')
y = y.values.ravel()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Load model
file_path = 'saved_models/'
xgb_best_fit_gs = joblib.load(file_path + 'xgb_best_model_under.pkl')

# optimal parameters
print(xgb_best_fit_gs.get_params())

# Show model performance 
y_test_pred_xgb_gs = xgb_best_fit_gs.predict(X_test)
print(classification_report(y_test, y_test_pred_xgb_gs)) 

# Confusion Matrix
cm_xgb = confusion_matrix(y_test, y_test_pred_xgb_gs)

# Normalize the confusion matrix
cm_xgb_norm = cm_xgb.astype('float') / cm_xgb.sum(axis=1)[:, np.newaxis]

# Create a custom colormap
cmap_custom = sns.diverging_palette(10, 150, as_cmap=True)

# Plot the confusion matrix 
plt.figure(figsize=(8, 6))
sns.heatmap(cm_xgb_norm, annot=True, cmap=cmap_custom, annot_kws={"size": 16},
            vmin=0, vmax=1)  
plt.title('Confusion Matrix - XGBoost')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# ROC curve
y_val_roc_xgb = xgb_best_fit_gs.predict_proba(X_test)[:, 1]

# Calculate ROC curve and AUC for text set
fpr_xgb, tpr_xgb, thresholds_xgb = roc_curve(y_test, y_val_roc_xgb)
roc_auc_xgb = auc(fpr_xgb, tpr_xgb)

# Plot ROC curve for validation set
plt.figure(figsize=(8, 6))
plt.plot(fpr_xgb, tpr_xgb, color='darkorange', lw=2, label='ROC curve (AUC = {:.2f})'.format(roc_auc_xgb))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('Specificity - FP Rate')
plt.ylabel('Sensitivity - TP Rate')
plt.title('ROC Curve - XGBoost')
plt.legend(loc='lower right')
plt.show()

# Feature importance
fi_xgb = xgb_best_fit_gs.feature_importances_
feature_names = X_train.columns

fi_df_xgb = pd.DataFrame({'Feature': feature_names, 'Importance': fi_xgb})
fi_df_xgb = fi_df_xgb.sort_values(by='Importance', ascending=True)

palette = sns.color_palette("Blues", n_colors=len(fi_df_xgb))

# Plot the feature importances
plt.figure(figsize=(10, 6))
plt.barh(fi_df_xgb['Feature'], fi_df_xgb['Importance'], color = palette)
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importance - XGBoost')
plt.show()







