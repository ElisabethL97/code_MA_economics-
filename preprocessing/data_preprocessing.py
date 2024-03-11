## Code for data preprocessing 

# Packages 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

# Data
file_path = "data/loan_final313.csv"
data = pd.read_csv(file_path)
print(data.shape)
data.head() # The dataset has 30 columns and 887379 rows.

# Data cleaning
data.info() # We can see that the dataset consists of variables with three different datatypes: float64, int64, and object.

# convert some columns to the correct data types
data['final_d'] = data['final_d'].astype(object)
data['annual_inc'] = data['annual_inc'].astype(float)
data['loan_amount'] = data['loan_amount'].astype(float)
data.info() # Now the columns have the correct datatypes.

# Check missing values
data.isnull().sum() # As we have a fictional dataset, no missing values apply.

##################################### Variable Selection ######################################################
###############################################################################################################

### Remove certain columns
print(data.shape)
data.head() # Data dimensions: 887379 rows, 30 columns. Each row is labeled.
data.info() # Three different dataypes are presented. Those different types are reasonable.

# Remove columns which will for sure not be used for modelling
data_prep = data
columns_remove = ['id', 'year', 'issue_d', 'final_d', 'total_pymnt', 'total_rec_prncp',
                  'recoveries', 'installment', 'region']
data_prep = data_prep.drop(columns=columns_remove)
data_prep.head()

# Remove columns with identical information
columns_remove = ['home_ownership', 'income_category', 'term', 'application_type',
                 'purpose', 'interest_payments', 'loan_condition', 'grade']
data_prep = data_prep.drop(columns=columns_remove)
print(data_prep.shape)
data_prep.head()

data_prep.info()

### Remove highly correlated variables
# Check correlation
plt.figure(figsize=(12, 10))
sns.heatmap(data_prep.corr(), annot=True, fmt="0.1f")
plt.title('Correlation Heatmap')
plt.show() # We can see that the majority of the variables are not correlated. Variables with a correlation of >0.75 will be removed.

# Remove highly correlated variables
corr_matrix = data_prep.corr()

high_corr_vars = set()

for i in range(len(corr_matrix.columns)):
    for j in range(i):
        if abs(corr_matrix.iloc[i, j]) >= 0.75:
            colname = corr_matrix.columns[i]
            high_corr_vars.add(colname)

data_prep = data_prep.drop(columns=high_corr_vars)
print(data_prep.shape)
data_prep.head()

# Check correlation again
plt.figure(figsize=(12, 10))
sns.heatmap(data_prep.corr(), annot=True, fmt="0.1f")
plt.title('Correlation Heatmap')
plt.show() # No variables show a correlation >= 0.75 anymore.

data_prep.info() # So far we have one dependent variable y (=loan_condition_cat) and 10 explanatory variables.

### Check outliers

# Vizualize outliers for float varaibles
for column in data_prep.select_dtypes(include=['float64']):
    sns.boxplot(x=data_prep[column])
    plt.title(f'Box Plot for {column}')
    plt.show()

# Check statitsics of float variables
data_prep.select_dtypes(include=['float64']).describe() # We can see there are indeed outliers in the data.



############################### Variable transformation ######################################################
##############################################################################################################

# Normalizing the float variables
data_prep_mod = data_prep
scaler = MinMaxScaler()
float_columns = data_prep_mod.select_dtypes(include=['float64']).columns.tolist()
data_prep_mod[float_columns] = scaler.fit_transform(data_prep_mod[float_columns])
data_prep_mod.select_dtypes(include=['float64']).describe()

# Create dependent and explanatory variables
y = data_prep_mod['loan_condition_cat']
X = data_prep_mod.drop(columns='loan_condition_cat')

X.info()

### Encode categorical variables
# Encode categorical variables
cat_columns = X.select_dtypes(include=['int64']).columns.tolist()
X_encoded = pd.get_dummies(X, columns=cat_columns)
bool_columns = X_encoded.select_dtypes(include=['bool']).columns.tolist()
X_encoded[bool_columns] = X_encoded[bool_columns].astype(int)
X_encoded.head()

X_encoded.info()

# Rename columns
X_encoded = X_encoded.rename(columns={'home_ownership_cat_1':'HO_RENT', 
                                      'home_ownership_cat_2': 'HO_OWN',
                                      'home_ownership_cat_3': 'HO_MORTGAGE',
                                      'home_ownership_cat_4': 'HO_OTHER',
                                      'home_ownership_cat_5': 'HO_NONE',
                                      'home_ownership_cat_6': 'HO_ANY', 
                                      'income_cat_1': 'Income_LOW',
                                      'income_cat_2': 'Income_MEDIUM',
                                      'income_cat_3': 'Income_HIGH',
                                      'term_cat_1': 'Term_36MONTHS',
                                      'term_cat_2': 'Term_60MONTHS', 
                                      'application_type_cat_1': 'Type_INDIVIDUAL',
                                      'application_type_cat_2': 'Type_JOINT', 
                                      'purpose_cat_1': 'Purpose_credit_card', 
                                      'purpose_cat_2': 'Purpose_car',
                                      'purpose_cat_3': 'Purpose_small_business',
                                      'purpose_cat_4': 'Purpose_other',
                                      'purpose_cat_5': 'Purpose_wedding',
                                      'purpose_cat_6': 'Purpose_debt_consolidation',
                                      'purpose_cat_7': 'Purpose_home_improvement',
                                      'purpose_cat_8': 'Purpose_major_purchase',
                                      'purpose_cat_9': 'Purpose_medical',
                                      'purpose_cat_10': 'Purpose_moving',
                                      'purpose_cat_11': 'Purpose_vacation',
                                      'purpose_cat_12': 'Purpose_house',
                                      'purpose_cat_13': 'Purpose_renewable_energy',
                                      'purpose_cat_14': 'Purpose_educational',
                                      'interest_payment_cat_2': 'Interest_HIGH', 
                                      'interest_payment_cat_1': 'Interest_LOW'})

print(X_encoded.info())
X_encoded.shape

### Reduce dimensions of dataset
# Dimension reduction
# Step 1 domain knowledge
X_mod = X_encoded
columns_remove = ['HO_OTHER', 'HO_ANY', 'HO_NONE', 'Type_JOINT', 'Type_INDIVIDUAL', 
                  'Purpose_medical', 'Purpose_moving', 'Purpose_vacation', 'Purpose_house', 'Purpose_wedding',
                  'Purpose_renewable_energy', 'Purpose_educational']
X_mod = X_mod.drop(columns=columns_remove)
X_mod.head()

X_mod.shape

# Check correlation again
plt.figure(figsize=(12, 10))
sns.heatmap(X_mod.corr(), annot=True, fmt="0.1f")
plt.title('Correlation Heatmap')
plt.show()

# Remove highly correlated variables
corr_matrix = X_mod.corr()

high_corr_vars = set()

for i in range(len(corr_matrix.columns)):
    for j in range(i):
        if abs(corr_matrix.iloc[i, j]) >= 0.75:
            colname = corr_matrix.columns[i]
            high_corr_vars.add(colname)

X_mod = X_mod.drop(columns=high_corr_vars)
print(X_mod.shape)
X_mod.head()

# Check final correlation
plt.figure(figsize=(12, 10))
sns.heatmap(X_mod.corr(), annot=True, fmt="0.1f")
plt.title('Correlation Heatmap')
plt.show()

print(X_mod.shape) # 17 explanatory variables will enter the models
print(y.shape)




############################# Handle imbalanced dataset ######################################################
##############################################################################################################

# Target Variable - loan condition category
count = data['loan_condition_cat'].value_counts()
share = count / len(data) * 100

plt.pie(share,
        labels=share.index,
        wedgeprops = {'linewidth' : 3, 'edgecolor' : 'white'},
        autopct='%1.1f%%',
        startangle=140,
        colors = sns.color_palette("Set2"))
plt.title('Distribution of target variable')
plt.show()

X = X_mod
X.shape

# Save preprocessed data 
X.to_csv('X_preprocessed.csv', index=False)
y.to_csv('y_preprocessed.csv', index=False)




