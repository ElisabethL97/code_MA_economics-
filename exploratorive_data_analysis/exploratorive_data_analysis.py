## Code for exploratorive data analysis

# Packages
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Data
data = pd.read_csv("loan_final313.csv")
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

########################### Exploratorive Data Analysis ######################################################
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

# Defaults per year
count = data.groupby(['year', 'loan_condition_cat']).size().unstack(fill_value=0)
share = count / len(data) * 100

ax = share.plot(kind='bar', 
                stacked=True,
                color=sns.color_palette("Set2"))
ax.set_ylabel('Share in percentage')
ax.set_xlabel('')
ax.set_title('Amount of loans per year')
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
plt.legend(title='Loan condition',
           labels=['0 - Good Loan', '1 - Bad Loan'])
plt.show()

# Employment Length
count = data.groupby(['emp_length_int', 'loan_condition_cat']).size().unstack(fill_value=0)
share = count / len(data) * 100
ax = share.plot(kind='bar',
                stacked=True,
                color=sns.color_palette("Set2"))
ax.set_ylabel('Share in percentage')
ax.set_xlabel('Employment length in years')
ax.set_title('Employment length')
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
plt.legend(title='Loan condition',
           labels=['0 - Good Loan', '1 - Bad Loan'])
plt.show()

# Home ownership status
count = data.groupby(['home_ownership', 'loan_condition_cat']).size().unstack(fill_value=0)
share = count / len(data) * 100
share_sorted = share.sum(axis=1).sort_values(ascending=False).index
share = share.loc[share_sorted]

ax = share.plot(kind='bar',
                stacked=True,
                color=sns.color_palette("Set2"))
ax.set_ylabel('Share in percentage')
ax.set_xlabel('')
ax.set_title('Home ownership status')
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
plt.legend(title='Loan condition',
           labels=['0 - Good Loan', '1 - Bad Loan'])
plt.show()

# Income category
count = data.groupby(['income_category', 'loan_condition_cat']).size().unstack(fill_value=0)
share = count / len(data) * 100
share_sorted = share.sum(axis=1).sort_values(ascending=False).index
share = share.loc[share_sorted]

ax = share.plot(kind='bar',
                stacked=True,
                color=sns.color_palette("Set2"))
ax.set_ylabel('Share in percentage')
ax.set_xlabel('')
ax.set_title('Income category')
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
plt.legend(title='Loan condition',
           labels=['0 - Good Loan', '1 - Bad Loan'])
plt.show()

# Term - loan duration
count = data.groupby(['term', 'loan_condition_cat']).size().unstack(fill_value=0)
share = count / len(data) * 100

ax = share.plot(kind='bar',
                stacked=True,
                color=sns.color_palette("Set2"))
ax.set_ylabel('Share in percentage')
ax.set_xlabel('')
ax.set_title('Loan duration')
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
plt.legend(title='Loan condition',
           labels=['0 - Good Loan', '1 - Bad Loan'])
plt.show()

# Application type
count = data.groupby(['application_type', 'loan_condition_cat']).size().unstack(fill_value=0)
share = count / len(data) * 100

ax = share.plot(kind='bar',
                stacked=True,
                color=sns.color_palette("Set2"))
ax.set_ylabel('Share in percentage')
ax.set_xlabel('')
ax.set_title('Application type')
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
plt.legend(title='Loan condition',
           labels=['0 - Good Loan', '1 - Bad Loan'])
plt.show()

# Purpose
count = data.groupby(['purpose', 'loan_condition_cat']).size().unstack(fill_value=0)
share = count / len(data) * 100
share_sorted = share.sum(axis=1).sort_values(ascending=True).index
share = share.loc[share_sorted]

ax = share.plot(kind='barh',
                stacked=True,
                color=sns.color_palette("Set2"))
ax.set_ylabel('')
ax.set_xlabel('Share in percentage')
ax.set_title('Purpose for issuing a loan')
plt.legend(title='Loan condition',
           labels=['0 - Good Loan', '1 - Bad Loan'], loc='lower right')
plt.show()

# Interest payments
count = data.groupby(['interest_payments', 'loan_condition_cat']).size().unstack(fill_value=0)
share = count / len(data) * 100

ax = share.plot(kind='bar',
                stacked=True,
                color=sns.color_palette("Set2"))
ax.set_ylabel('Share in percentage')
ax.set_xlabel('')
ax.set_title('Interest payment categories')
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
plt.legend(title='Loan condition',
           labels=['0 - Good Loan', '1 - Bad Loan'], loc='lower right')
plt.show()

# Grade
count = data.groupby(['grade', 'loan_condition_cat']).size().unstack(fill_value=0)
share = count / len(data) * 100

ax = share.plot(kind='bar',
                stacked=True,
                color=sns.color_palette("Set2"))
ax.set_ylabel('Share in percentage')
ax.set_xlabel('')
ax.set_title('Grade categories')
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
plt.legend(title='Loan condition',
           labels=['0 - Good Loan', '1 - Bad Loan'],
           loc='upper right')
plt.show()

# Regions
count = data.groupby(['region', 'loan_condition_cat']).size().unstack(fill_value=0)
share = count / len(data) * 100
share_sorted = share.sum(axis=1).sort_values(ascending=False).index
share = share.loc[share_sorted]

ax = share.plot(kind='bar',
                stacked=True,
                color=sns.color_palette("Set2"))
ax.set_ylabel('Share in percentage')
ax.set_xlabel('')
ax.set_title('Region')
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
plt.legend(title='Loan condition',
           labels=['0 - Good Loan', '1 - Bad Loan'],
           loc='upper right')
plt.show()


data['annual_inc'].describe()

# Annual income
good_loan = data[data['loan_condition_cat'] == 0]['annual_inc']
bad_loan = data[data['loan_condition_cat'] == 1]['annual_inc']

plt.hist([good_loan, bad_loan],
         bins=15,
         stacked=True,
         color=['#66c2a5', '#fc8d62'],
         rwidth=0.9,
         label=['Good Loan', 'Bad Loan'])
plt.xlabel('Annual income')
plt.ylabel('Frequency')
plt.title('Annual income distribution')
plt.legend()
plt.show() # We can see a problem with the annual income column. High outliers are assumed.

# Loan amount
good_loan = data[data['loan_condition_cat'] == 0]['loan_amount']
bad_loan = data[data['loan_condition_cat'] == 1]['loan_amount']

plt.hist([good_loan, bad_loan],
         bins=15,
         stacked=True,
         color=['#66c2a5', '#fc8d62'],
         rwidth=0.9,
         label=['Good Loan', 'Bad Loan'])
plt.xlabel('Loan Amount')
plt.ylabel('Frequency')
plt.title('Loan amount distribution')
plt.legend()
plt.show()

# Interest Rate
good_loan = data[data['loan_condition_cat'] == 0]['interest_rate']
bad_loan = data[data['loan_condition_cat'] == 1]['interest_rate']

plt.hist([good_loan, bad_loan],
         bins=15,
         stacked=True,
         color=['#66c2a5', '#fc8d62'],
         rwidth=0.9,
         label=['Good Loan', 'Bad Loan'])
plt.xlabel('Interest rate')
plt.ylabel('Frequency')
plt.title('Interest rate distribution')
plt.legend()
plt.show()

# Dti - ratio of monthly debt payments to annual income
good_loan = data[data['loan_condition_cat'] == 0]['dti']
bad_loan = data[data['loan_condition_cat'] == 1]['dti']

plt.hist([good_loan, bad_loan],
         bins=30,
         stacked=True,
         color=['#66c2a5', '#fc8d62'],
         rwidth=0.9,
         label=['Good Loan', 'Bad Loan'])
plt.xlabel('dti')
plt.ylabel('Frequency')
plt.title('Ratio of monthly debt payments to annual income distribution')
plt.legend()
plt.show() # We can see a problem with the dti column. High outliers are assumed.

# Total payment
good_loan = data[data['loan_condition_cat'] == 0]['total_pymnt']
bad_loan = data[data['loan_condition_cat'] == 1]['total_pymnt']

plt.hist([good_loan, bad_loan],
         bins=15,
         stacked=True,
         color=['#66c2a5', '#fc8d62'],
         rwidth=0.9,
         label=['Good Loan', 'Bad Loan'])
plt.xlabel('Total payment')
plt.ylabel('Frequency')
plt.title('Total payment distribution')
plt.legend()
plt.show()

# Total rec prncp
good_loan = data[data['loan_condition_cat'] == 0]['total_rec_prncp']
bad_loan = data[data['loan_condition_cat'] == 1]['total_rec_prncp']

plt.hist([good_loan, bad_loan],
         bins=15,
         stacked=True,
         color=['#66c2a5', '#fc8d62'],
         rwidth=0.9,
         label=['Good Loan', 'Bad Loan'])
plt.xlabel('Total rec prncp')
plt.ylabel('Frequency')
plt.title('Total rec prncp distribution')
plt.legend()
plt.show()

# Recoveries
good_loan = data[data['loan_condition_cat'] == 0]['recoveries']
bad_loan = data[data['loan_condition_cat'] == 1]['recoveries']

plt.hist([good_loan, bad_loan],
         bins=15,
         stacked=True,
         color=['#66c2a5', '#fc8d62'],
         rwidth=0.9,
         label=['Good Loan', 'Bad Loan'])
plt.xlabel('Recoveries')
plt.ylabel('Frequency')
plt.title('Recoveries distribution')
plt.legend()
plt.show() # We can see a problem with the recoveries column. High outliers are assumed.

# Installments
good_loan = data[data['loan_condition_cat'] == 0]['installment']
bad_loan = data[data['loan_condition_cat'] == 1]['installment']

plt.hist([good_loan, bad_loan],
         bins=15,
         stacked=True,
         color=['#66c2a5', '#fc8d62'],
         rwidth=0.9,
         label=['Good Loan', 'Bad Loan'])
plt.xlabel('Installments')
plt.ylabel('Frequency')
plt.title('Installments distribution')
plt.legend()
plt.show()

## Annual Income by income category
sns.boxplot(data=data,
            x='income_category',
            y='annual_inc',
            hue='loan_condition',
            palette="Set2")
plt.ylim(0, 600000)
plt.ylabel('Annual income')
plt.xlabel('')
plt.title('Annual income by income category')
plt.legend()
plt.show() # We can see high outliers for the high income class.
