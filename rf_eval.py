## Code for Random Forest model evaluation

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

# Split the data into training, and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Load model
file_path = 'saved_models/'
rf_best_fit_gs = joblib.load(file_path + 'rf_best_model.pkl')

# optimal parameters
print(rf_best_fit_gs.get_params())

# Show model performance 
y_test_pred_rf_gs = rf_best_fit_gs.predict(X_test)
print(classification_report(y_test, y_test_pred_rf_gs)) 

# Confusion Matrix
cm_rf = confusion_matrix(y_test, y_test_pred_rf_gs)

# Normalize the confusion matrix
cm_rf_norm = cm_rf.astype('float') / cm_rf.sum(axis=1)[:, np.newaxis]

# Create a custom colormap
cmap_custom = sns.diverging_palette(10, 150, as_cmap=True)

# Plot the confusion matrix 
plt.figure(figsize=(8, 6))
sns.heatmap(cm_rf_norm, annot=True, cmap=cmap_custom, annot_kws={"size": 16},
            vmin=0, vmax=1)  
plt.title('Confusion Matrix - Random Forest')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# ROC curve
y_val_roc_rf = rf_best_fit_gs.predict_proba(X_test)[:, 1]

# Calculate ROC curve and AUC for validation set
fpr_rf, tpr_rf, thresholds_rf = roc_curve(y_test, y_val_roc_rf)
roc_auc_rf = auc(fpr_rf, tpr_rf)

# Plot ROC curve for validation set
plt.figure(figsize=(8, 6))
plt.plot(fpr_rf, tpr_rf, color='darkorange', lw=2, label='ROC curve (AUC = {:.2f})'.format(roc_auc_rf))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('Specificity - FP Rate')
plt.ylabel('Sensitivity - TP Rate')
plt.title('ROC Curve - Random Forest')
plt.legend(loc='lower right')
plt.show()


# Feature importance
fi_rf = rf_best_fit_gs.feature_importances_
feature_names = X_train.columns

fi_df_rf = pd.DataFrame({'Feature': feature_names, 'Importance': fi_rf})
fi_df_rf = fi_df_rf.sort_values(by='Importance', ascending=True)

palette = sns.color_palette("Blues", n_colors=len(fi_df_rf))

# Plot the feature importances
plt.figure(figsize=(10, 6))
plt.barh(fi_df_rf['Feature'], fi_df_rf['Importance'], color = palette)
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importance - Random Forest')
plt.show()



