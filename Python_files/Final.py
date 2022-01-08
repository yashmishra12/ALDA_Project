#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import scipy
import seaborn as sns

import missingno as msno


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import warnings

warnings.filterwarnings("ignore")

import sklearn.preprocessing
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import recall_score, accuracy_score, confusion_matrix, f1_score
from sklearn.metrics import precision_score, auc, roc_auc_score, roc_curve, precision_recall_curve

from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

from sklearn.linear_model import Ridge, Lasso
from sklearn.feature_selection import RFE
from xgboost import XGBClassifier

from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm
from sklearn import metrics

from sklearn.metrics import mean_squared_error

from sklearn.ensemble import GradientBoostingClassifier  #GBM algorithm
# Training the model on the train data
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm
from sklearn import metrics



from scipy.stats.mstats import winsorize

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler 

from sklearn.decomposition import PCA


# In[ ]:


# !pip install lazypredict


# In[ ]:


# !pip uninstall scikit-learn -y
# # 


# In[ ]:


# !pip install scikit-learn==0.23.1


# In[ ]:


from lazypredict.Supervised import LazyClassifier
from sklearn.utils import shuffle


# ### Reading CSV files

# In[ ]:


df_2014 = pd.read_csv("2014_Financial_Data.csv")
df_2015 = pd.read_csv("2015_Financial_Data.csv")
df_2016 = pd.read_csv("2016_Financial_Data.csv")
df_2017 = pd.read_csv("2017_Financial_Data.csv")
df_2018 = pd.read_csv("2018_Financial_Data.csv")


# In[ ]:


nasdaq = pd.read_csv("nasdaq.csv")


# In[ ]:


nasdaq.shape


# In[ ]:


nasdaq.columns


# In[ ]:


nasdaq.head()


# In[ ]:


nasdaq.drop('Name', axis=1, inplace=True)


# In[ ]:


df_2014.shape


# In[ ]:


df_2015.shape


# In[ ]:


df_2016.shape


# In[ ]:


df_2017.shape


# In[ ]:


df_2018.shape


# In[ ]:


df_2014['Year'] = 2014
df_2015['Year'] = 2015
df_2016['Year'] = 2016
df_2017['Year'] = 2017
df_2018['Year'] = 2018


# In[ ]:


df_2014.columns


# In[ ]:


df_2014.rename(columns={'2015 PRICE VAR [%]':'Next_Year_Price_Var[%]'}, inplace=True)
df_2015.rename(columns={'2016 PRICE VAR [%]':'Next_Year_Price_Var[%]'}, inplace=True)
df_2016.rename(columns={'2017 PRICE VAR [%]':'Next_Year_Price_Var[%]'}, inplace=True)
df_2017.rename(columns={'2018 PRICE VAR [%]':'Next_Year_Price_Var[%]'}, inplace=True)
df_2018.rename(columns={'2019 PRICE VAR [%]':'Next_Year_Price_Var[%]'}, inplace=True)


# In[ ]:


df = pd.concat([df_2014, df_2015, df_2016, df_2017, df_2018], axis = 0)


# In[ ]:


df.shape


# In[ ]:


df.columns


# In[ ]:


df.head()


# In[ ]:


df.rename(columns={"Unnamed: 0": "Symbol"}, inplace=True)
# df_2014.rename(columns={'2015 PRICE VAR [%]':'Next_Year_Price_Var[%]'}, inplace=True)


# In[ ]:


df = pd.merge(df, nasdaq, how="inner", on="Symbol")


# In[ ]:


df.shape


# In[ ]:


df.head()


# In[ ]:


df.Country.value_counts()


# In[ ]:


df.Country.isnull().sum()


# In[ ]:





# In[ ]:


#Next_Year_Price_Var[%] +ve ---> class = 1, if -ve -----> class = 0
df.drop('Next_Year_Price_Var[%]', axis=1, inplace=True)


# In[ ]:


df.rename(columns={"Symbol":"Name"}, inplace=True)


# In[ ]:


df.Name.nunique()


# In[ ]:


df.shape


# In[ ]:


df.info


# In[ ]:


df.describe()


# In[ ]:


## Removing "Year" because our future companies will have different years and it should not affect our final call
df.drop("Year", axis=1, inplace=True)


# In[ ]:


## Removing "Name" because our future companies will have different Name and it should not affect our final call
df.drop("Name", axis=1, inplace=True)


# In[ ]:


df.head(2)


# In[ ]:


df.loc[(df.Country=="United States")].shape


# In[ ]:


df.loc[(df.Country=="United States")]


# In[ ]:


# Selecting only United States


# In[ ]:


df.Country.value_counts()


# In[ ]:


df = df.loc[(df.Country=="United States")]


# In[ ]:


df.drop('Country', axis=1, inplace=True)


# ### Understanding Null Value Distribution

# In[ ]:


df.isnull().sum().sort_values(ascending=False)


# In[ ]:


(df.isnull().sum() * 100 / len(df)).sort_values(ascending=False)


# In[ ]:


(df.isnull().sum() * 100 / len(df)).sort_values(ascending=True).head(15)


# In[ ]:


# all cols have some null values
len(df.isnull().any())


# In[ ]:


# No columns with all null values
df.columns[df.isnull().all()]


# In[ ]:


# Defining a funtion to add the count/frequency values as annotation to histogram.
def annotate_graph(ax):
    for bar in ax.patches:         
        ax.annotate(format((bar.get_height()), '.0f'),                    
                    (bar.get_x() + bar.get_width() / 2,  bar.get_height()),                    
                    ha='center', va='center',                    
                    size=10, xytext=(0, 8),                    
                    textcoords='offset points')
    return ax


# In[ ]:


# Plotting histogram for the dataframe and columns having null values.
plt.figure(figsize=(28,10))

ax = sns.histplot(round((df.isnull().sum()/len(df.index) * 100).sort_values(ascending=False), 2))
ax = annotate_graph(ax)

ax.set(xticks=np.arange(0,101))
ax.set(xlabel='Null value percentage', ylabel='Count of columns with null values')
sns.despine()
plt.tight_layout()


# In[ ]:


msno.matrix(df)


# In[ ]:


# defining a function to get more than cutoff percent missing value

def get_missing_value_percentage(cutoff):
    y = pd.DataFrame( round((df.isnull().sum()/len(df.index) * 100).sort_values(ascending=False), 2))
    y.rename(columns={0:"Percentage"}, inplace=True)
    y2 = y[y.Percentage>cutoff]
    return y2


# In[ ]:


# get columns with more than 70% missing values
greater_than_70 = get_missing_value_percentage(70)


# In[ ]:


len(greater_than_70)


# In[ ]:


greater_than_70


# In[ ]:


# get columns with more than 50% missing values
greater_than_50 = get_missing_value_percentage(50)


# In[ ]:


len(greater_than_50)


# In[ ]:


greater_than_50


# In[ ]:


# get columns with more than 20% missing values
greater_than_20 = get_missing_value_percentage(20)
greater_than_20


# ### Removing Null Values

# In[ ]:


# function to drop cols which have more than 20% null values

def remove_cols_with_nulls (df, threshold):
    myCol = list(df.columns)
    for col in myCol: 
        percentage = (df[col].isnull().sum()/len(df[col]))*100
        if percentage>threshold:
            df.drop(col, axis=1, inplace=True)


# In[ ]:


df.shape


# In[ ]:


remove_cols_with_nulls(df, 20)


# In[ ]:


df.shape


# In[ ]:


len(df.columns[(df.isnull().any())])


# In[ ]:


# Deleting rows with any null value
df.dropna(how='all',axis=0, inplace=True) 


# In[ ]:


# Therefore, there is no row will all NULL values
df.shape


# In[ ]:


# Deleting rows with any null value
df.dropna(how='any',axis=0, inplace=True) 


# In[ ]:


df.shape


# In[ ]:


msno.matrix(df)


# In[ ]:


df.isnull().any().sum()


# In[ ]:


df.columns


# In[ ]:


df.select_dtypes('number')


# In[ ]:


df['R&D Expenses'].value_counts()


# In[ ]:


df.shape[0]


# In[ ]:


df['R&D Expenses'].value_counts(normalize=True).sort_values(ascending=False)[0]


# In[ ]:


# more than 50% value of R&D expense is 0. Remove numeric cols with dominant values


# In[ ]:


df.operatingProfitMargin.value_counts().sort_values(ascending=False).iloc[0]


# In[ ]:


#Code goes to except block when single value is met ---> which is SERIES and you cannot do simple indexing in SERIES


# In[ ]:


counter = 0
for col in list(df.select_dtypes('number').columns):
    try:  
        val = df[col].value_counts(normalize=True).sort_values(ascending=False)[0]
        if(val>0.5):
            df.drop(col, axis=1, inplace=True)
            counter = counter+1
    except:
        val = df[col].value_counts(normalize=True).sort_values(ascending=False).iloc[0]
        if(val>0.5):
            df.drop(col, axis=1, inplace=True)
            counter = counter+1
        
    
print("Total Columns Deleted = ",counter)


# In[ ]:


df.shape


# In[ ]:


df.columns


# In[ ]:


df.select_dtypes(include='number')


# In[ ]:


df.select_dtypes(include='object')


# In[ ]:


df.select_dtypes(include='number').shape


# In[ ]:


df.select_dtypes(include='object').shape


# In[ ]:


df.select_dtypes(include='category').shape


# In[ ]:


df.shape


# In[ ]:


# 155 cols ----> 154 number, 1 Object


# In[ ]:


df.Sector.value_counts()


# In[ ]:


sector_list = list(df.Sector.unique())


# In[ ]:


sector_list


# In[ ]:


pd.get_dummies(df.Sector, drop_first=True)


# In[ ]:


Sector_status = pd.get_dummies(df.Sector, drop_first=True)

#Adding the result to the original housing dataframe

df = pd.concat([df, Sector_status], axis=1)


# In[ ]:


# Droppig Sector Column as we are done with 
df.drop("Sector", axis=1, inplace=True)


# In[ ]:


df.shape


# In[ ]:


df.Energy.value_counts()


# In[ ]:


# Sector Column will be dropped after Exploratory Data Analysis


# In[ ]:


df.head()


# # Removing columns with single value

# In[ ]:


def removeSingleValue (col):
    length = len(df[col].value_counts())
    if (length<2):
        print(col)
        df.drop(col, axis=1, inplace=True)


# In[ ]:


for col in df.columns:
    removeSingleValue(col)


# In[ ]:


num_col = list(df.dtypes[df.dtypes !='object'].index)


# In[ ]:


len(num_col)


# In[ ]:


df.shape


# ### Duplicate Row Checker

# In[ ]:


df.duplicated().sum()


# In[ ]:


# Moving "Class" Column to end
df['Result'] = df.Class
df.drop("Class", axis=1, inplace=True)
df = df.rename(columns={"Result":"Class"})


# In[ ]:





# In[ ]:


df.head()


# # Exploratory Data Analysis

# In[ ]:


# Get number of positve and negative examples
pos = df[df["Class"] == 1].shape[0]
neg = df[df["Class"] == 0].shape[0]

print("Profitable Companies = ",pos)
print("Lossy Companies = ",neg)


# In[ ]:


# Plotting the profitable vs non-profitable customers.
plt.figure(figsize = (12, 4))

length = len(df)
ax = sns.barplot(x = ['Profitable','Lossy'], y = [pos/length * 100,neg/length * 100])
plt.xlabel("Class", labelpad = 15)
plt.ylabel('Percentage Rate', labelpad = 10)

# Call Custom Function
annotate_graph(ax)


sns.despine()
plt.tight_layout()
plt.show()


# In[ ]:


# Therefore, the output class is balanced


# In[ ]:


corr_df = pd.DataFrame(df.corr()['Class'].sort_values(ascending = False))


# In[ ]:


positive_corr = corr_df.loc[(corr_df.Class>0) & (corr_df.Class<1)]


# In[ ]:


negative_corr = corr_df.loc[corr_df.Class<0]


# In[ ]:


# Creating bar chart for showing co-relation of all variables with Class.
plt.figure(figsize=(20,10))
positive_corr.Class.plot(kind = 'bar')
plt.show()


# In[ ]:


positive_corr=positive_corr.reset_index()


# In[ ]:


positive_corr


# In[ ]:


positive_corr = positive_corr.rename(columns={"index":"col_name"})


# In[ ]:


positive_corr


# In[ ]:


top_pos_index = list(positive_corr.col_name[0:6])


# In[ ]:


top_pos_index


# In[ ]:


top_pos_index.remove("Utilities")
# top_pos_index.remove("Financial Services")


# In[ ]:


printThis = df[top_pos_index]
printThis['Class'] = df.Class


# In[ ]:


# Plotting most positively related cols

plt.figure(figsize=(20,12))
sns.pairplot(printThis, hue="Class")
sns.despine()
plt.tight_layout()


plt.plot()


# In[ ]:


# Creating bar chart for showing co-relation of all variables with Class.
plt.figure(figsize=(20,10))
negative_corr.Class.plot(kind = 'bar')
plt.show()


# In[ ]:





# In[ ]:


negative_corr=negative_corr.reset_index()


# In[ ]:


negative_corr


# In[ ]:


negative_corr = negative_corr.rename(columns={"index":"col_name"})


# In[ ]:


negative_corr


# In[ ]:


top_neg_index = list(negative_corr.col_name[0:4])


# In[ ]:


top_neg_index


# In[ ]:


printThis2 = df[top_neg_index]
printThis2['Class'] = df.Class


# In[ ]:


# Plotting most positively related cols

plt.figure(figsize=(20,12))
sns.pairplot(printThis2, hue="Class")
sns.despine()
plt.tight_layout()


plt.plot()


# In[ ]:


df.columns


# In[ ]:


plotHelper = pd.DataFrame()
col_name = []
mean = []
std = []
for col in df.select_dtypes(include = np.number).columns.to_list():
    col_name.append(col)
    mean.append(df[col].mean())
    std.append(df[col].std())


# In[ ]:


plotHelper['col_name']=col_name
plotHelper['mean']=mean
plotHelper['std']=std


# In[ ]:


plotHelper


# In[ ]:


plotHelper = plotHelper.loc[~(plotHelper.col_name=="Class")]


# In[ ]:


plotHelper


# In[ ]:


plotHelper.sort_values(by="mean", ascending=True)


# In[ ]:


top_mean = list(plotHelper.sort_values(by="mean",ascending=False).col_name[0:5])


# In[ ]:


bottom_mean = list(plotHelper.sort_values(by="mean", ascending=True).col_name[0:5])


# In[ ]:


top_std = list(plotHelper.sort_values(by="std",ascending=False).col_name[0:5])


# In[ ]:


bottom_std = list(plotHelper.sort_values(by="std",ascending=True).col_name[0:5])


# In[ ]:


top_mean


# In[ ]:


# Plotting cols with highest mean

plt.figure(figsize=(20,12))
sns.pairplot(df[top_mean],diag_kind='kde')
sns.despine()
plt.tight_layout()


plt.plot()


# In[ ]:


# Plotting with lowest mean

plt.figure(figsize=(20,12))
sns.pairplot(df[bottom_mean],diag_kind='kde')
sns.despine()
plt.tight_layout()


plt.plot()


# In[ ]:


# Plotting cols with highest SD

plt.figure(figsize=(20,12))
sns.pairplot(df[top_std],diag_kind='kde')
sns.despine()
plt.tight_layout()


plt.plot()


# In[ ]:


# Plotting cols with lowest SD

plt.figure(figsize=(20,12))
sns.pairplot(df[bottom_std],diag_kind='kde')
sns.despine()
plt.tight_layout()


plt.plot()


# In[ ]:


top_mean


# In[ ]:


fig = plt.figure(figsize=(20,12))

ax0=fig.add_subplot(2,2,1)
ax1=fig.add_subplot(2,2,2)
ax2=fig.add_subplot(2,2,3)
ax3=fig.add_subplot(2,2,4)

sns.boxplot(df[top_mean[0]], data=df, ax=ax0)
sns.boxplot(df[top_mean[1]], data=df, ax=ax1)
sns.boxplot(df[top_mean[2]], data=df, ax=ax2)
sns.boxplot(df[top_mean[3]], data=df, ax=ax3)

sns.despine()
plt.tight_layout()
plt.plot()


# In[ ]:


fig = plt.figure(figsize=(20,12))

ax0=fig.add_subplot(2,2,1)
ax1=fig.add_subplot(2,2,2)
ax2=fig.add_subplot(2,2,3)
ax3=fig.add_subplot(2,2,4)

sns.boxplot(df[bottom_mean[0]], data=df, ax=ax0)
sns.boxplot(df[bottom_mean[1]], data=df, ax=ax1)
sns.boxplot(df[bottom_mean[2]], data=df, ax=ax2)
sns.boxplot(df[bottom_mean[3]], data=df, ax=ax3)

sns.despine()
plt.tight_layout()
plt.plot()


# In[ ]:


fig = plt.figure(figsize=(20,12))

ax0=fig.add_subplot(2,2,1)
ax1=fig.add_subplot(2,2,2)
ax2=fig.add_subplot(2,2,3)
ax3=fig.add_subplot(2,2,4)

sns.boxplot(df[top_std[0]], data=df, ax=ax0)
sns.boxplot(df[top_std[1]], data=df, ax=ax1)
sns.boxplot(df[top_std[2]], data=df, ax=ax2)
sns.boxplot(df[top_std[3]], data=df, ax=ax3)

sns.despine()
plt.tight_layout()
plt.plot()


# In[ ]:


fig = plt.figure(figsize=(20,12))

ax0=fig.add_subplot(2,2,1)
ax1=fig.add_subplot(2,2,2)
ax2=fig.add_subplot(2,2,3)
ax3=fig.add_subplot(2,2,4)

sns.boxplot(df[bottom_std[0]], data=df, ax=ax0)
sns.boxplot(df[bottom_std[1]], data=df, ax=ax1)
sns.boxplot(df[bottom_std[2]], data=df, ax=ax2)
sns.boxplot(df[bottom_std[3]], data=df, ax=ax3)

sns.despine()
plt.tight_layout()
plt.plot()


# In[ ]:


df.corr()['Class'].sort_values(ascending=False).head(20)


# In[ ]:


df.corr()['Class'].sort_values(ascending=False).tail(20)


# In[ ]:


list(df.corr()['Class'].sort_values(ascending=False).index[1:5])


# In[ ]:


hig_pos_corr = list(df.corr()['Class'].sort_values(ascending=False).index[1:5])

fig = plt.figure(figsize=(20,12))

ax0=fig.add_subplot(2,2,1)
ax1=fig.add_subplot(2,2,2)
ax2=fig.add_subplot(2,2,3)
ax3=fig.add_subplot(2,2,4)

sns.boxplot(df[hig_pos_corr[0]], hue="Class", data=df, ax=ax0)
sns.boxplot(df[hig_pos_corr[1]], hue="Class", data=df, ax=ax1)
sns.boxplot(df[hig_pos_corr[2]], hue="Class", data=df, ax=ax2)
sns.boxplot(df[hig_pos_corr[3]], hue="Class", data=df, ax=ax3)

sns.despine()
plt.tight_layout()
plt.plot()


# In[ ]:


hig_neg_corr = list(df.corr()['Class'].sort_values(ascending=True).index[0:7])


# In[ ]:


hig_neg_corr


# In[ ]:


hig_neg_corr.remove('Energy')


# In[ ]:


hig_neg_corr.remove('Healthcare')


# In[ ]:


# hig_neg_corr = list(df.corr()['Class'].sort_values(ascending=True).index[0:5])

fig = plt.figure(figsize=(20,12))

ax0=fig.add_subplot(2,2,1)
ax1=fig.add_subplot(2,2,2)
ax2=fig.add_subplot(2,2,3)
ax3=fig.add_subplot(2,2,4)

sns.boxplot(df[hig_neg_corr[0]], hue="Class", data=df, ax=ax0)
sns.boxplot(df[hig_neg_corr[1]], hue="Class", data=df, ax=ax1)
sns.boxplot(df[hig_neg_corr[2]], hue="Class", data=df, ax=ax2)
sns.boxplot(df[hig_neg_corr[3]], hue="Class", data=df, ax=ax3)

sns.despine()
plt.tight_layout()
plt.plot()


# In[ ]:


# Plotting the co-relation matrix for the data frame.
plt.figure(figsize = (25, 20))

ax=sns.heatmap(df.corr(),xticklabels=True, yticklabels=True)

# sns.despine()
plt.tight_layout()

plt.show()


# In[ ]:



# Plotting the co-relation matrix for the data frame.
plt.figure(figsize = (25, 10))

ax=sns.heatmap(df.corr().iloc[::-1],xticklabels=True, yticklabels=True)
ax.set_ylim(1,0)
# ax.set_xlim(5,0)
# sns.despine()
plt.tight_layout()

plt.show()


# In[ ]:


sector_list.remove("Basic Materials")


# In[ ]:


df[sector_list]


# In[ ]:


len(sector_list)


# In[ ]:


# Univariate Plot Analysis of Ordered categorical variables vs Percentage Rate
counter = 1

plt.figure(figsize=(20, 12))
for col_list in sector_list:
    df1 = df.loc[df[col_list]==1]
    series = round(((df[col_list].value_counts()) / (len(df[col_list])) * 100),
                   2)
    meraY = df.loc[df[col_list]==1].Class.value_counts().values

    plt.subplot(2, 5, counter)
    ax = sns.barplot(x=series.index,
                     y=meraY,
                     order=series.sort_index().index)
    plt.xlabel(col_list)
    plt.ylabel('Total Companies')

    annotate_graph(ax)
    counter += 1

sns.despine()
plt.tight_layout()
plt.show()


# In[ ]:





# In[ ]:





# # Outlier Treatment

# In[ ]:


df.head()


# ## Method 1 Standard Deviation Method
# 
# Three standard deviations from the mean is a common cut-off in practice for identifying outliers in a Gaussian or Gaussian-like distribution. For smaller samples of data, perhaps a value of 2 standard deviations (95%) can be used, and for larger samples, perhaps a value of 4 standard deviations (99.9%) can be used.

# In[ ]:


df.shape


# In[ ]:


# Extracting numerical columns from the telecom_df data frame.
numerical_cols = df.select_dtypes(include = np.number).columns.to_list()


# In[ ]:


# calculate summary statistics
data = df[numerical_cols]
data_mean, data_std = np.mean(data), np.std(data)
# identify outliers
cut_off = data_std * 2.5
lower, upper = data_mean - cut_off, data_mean + cut_off
# identify outliers
outliers = df[((df < lower) | (df > upper)).any(axis=1)]
print('Number of identified outliers: %d' % len(outliers))


# In[ ]:


# remove outliers
outliers_removed = df[~((df < lower) | (df > upper)).any(axis=1)]
print('Non-outlier observations: %d' % len(outliers_removed))


# ## Method 2 IQR method
# 
# The IQR can be used to identify outliers by defining limits on the sample values that are a factor k of the IQR below the 25th percentile or above the 75th percentile. The common value for the factor k is the value 1.5.

# In[ ]:


Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
print(IQR)


# In[ ]:


outliers_removed_IQR = df[~((df < (Q1 - 1.5 * IQR)) |(df > (Q3 + 1.5 * IQR))).any(axis=1)]
print('Non-outlier observations: %d' % len(outliers_removed_IQR))


# ## Method 3:  99-1 percentile method

# In[ ]:


Q1 = df.quantile(0.01)
Q3 = df.quantile(0.99)
IQR = Q3 - Q1
print(IQR)


# In[ ]:


outliers_removed_IQR = df[~((df < (Q1 - 1.5 * IQR)) |(df > (Q3 + 1.5 * IQR))).any(axis=1)]
print('Non-outlier observations: %d' % len(outliers_removed_IQR))


# In[ ]:


### We choose 99-1 percentile method for outlier treatment
df_99_1 = df[~((df < (Q1 - 1.5 * IQR)) |(df > (Q3 + 1.5 * IQR))).any(axis=1)]


# In[ ]:


df_99_1.shape


# ## Method 4:  95-5 percentile method

# In[ ]:


Q1 = df.quantile(0.05)
Q3 = df.quantile(0.95)
IQR = Q3 - Q1


# In[ ]:


outliers_removed_IQR = df[~((df < (Q1 - 1.5 * IQR)) |(df > (Q3 + 1.5 * IQR))).any(axis=1)]
print('Non-outlier observations: %d' % len(outliers_removed_IQR))


# In[ ]:


### We choose 95-5 percentile method for outlier treatment
df_95_5 = df[~((df < (Q1 - 1.5 * IQR)) |(df > (Q3 + 1.5 * IQR))).any(axis=1)]


# In[ ]:


df_95_5.shape


# # Observation
# 
# - Percentile method leads to poor performance later.
# 
# - Forced to go with 2.5 times the STD method after trial and error even though we are losing out on a lot of data

# In[ ]:


df = outliers_removed


# In[ ]:


df.reset_index(inplace=True)


# In[ ]:


df.drop(['index'], axis=1, inplace=True)


# In[ ]:


from sklearn.ensemble import IsolationForest


clf = IsolationForest()
preds = clf.fit_predict(df)


# In[ ]:


len(preds)


# In[ ]:


len(df)


# In[ ]:


for i in range(0,len(preds)):
    if preds[i]==-1:
        df.drop(i, axis=0, inplace=True)


# In[ ]:


len(df)


# ### Observation
# 
# - Isolation Forest Outlier Detection method further removed 4 data points

# In[ ]:





# # Lazy Classification Package

# In[ ]:


X = df.drop(['Class'], axis = 1)
y = df[['Class']]

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.8, test_size = 0.2, random_state = 100)

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


# In[ ]:


clf = LazyClassifier(verbose=1000,ignore_warnings=True, custom_metric=None)
models,predictions = clf.fit(X_train, X_test, y_train, y_test)


# In[ ]:


models.sort_values(by=["Accuracy", "F1 Score"], ascending=False)


# ## Technology Sector

# In[ ]:


dfTech = df.loc[df.Technology==1]

X = dfTech.drop(['Class'], axis = 1)
y = dfTech[['Class']]

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.8, test_size = 0.2, random_state = 100)


clf = LazyClassifier(verbose=1000,ignore_warnings=True, custom_metric=None)


# In[ ]:


models,predictions = clf.fit(X_train, X_test, y_train, y_test)


# In[ ]:


models.sort_values(by=["Accuracy", "F1 Score"], ascending=False)


# ## Industrial Sector

# In[ ]:


dfIndustrials = df.loc[df.Industrials==1]
X = dfIndustrials.drop(['Class'], axis = 1)
y = dfIndustrials[['Class']]

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.8, test_size = 0.2, random_state = 100)


# In[ ]:


clf = LazyClassifier(verbose=1000,ignore_warnings=True, custom_metric=None)
models,predictions = clf.fit(X_train, X_test, y_train, y_test)


# In[ ]:


models.sort_values(by=["Accuracy", "F1 Score"], ascending=False)


# ## HealthCare Sector

# In[ ]:


dfHealthcare = df.loc[df.Healthcare==1]


# In[ ]:


X = dfHealthcare.drop(['Class'], axis = 1)
y = dfHealthcare[['Class']]

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.8, test_size = 0.2, random_state = 100)


# In[ ]:


clf = LazyClassifier(verbose=1000,ignore_warnings=True, custom_metric=None)
models,predictions = clf.fit(X_train, X_test, y_train, y_test)


# In[ ]:


models.sort_values(by=["Accuracy", "F1 Score"], ascending=False)


# ## Financial Sector

# In[ ]:


dfFinancialServices = df.loc[df['Financial Services']==1]
X = dfFinancialServices.drop(['Class'], axis = 1)
y = dfFinancialServices[['Class']]

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.8, test_size = 0.2, random_state = 100)
clf = LazyClassifier(verbose=1000,ignore_warnings=True, custom_metric=None)
models,predictions = clf.fit(X_train, X_test, y_train, y_test)


# In[ ]:


models.sort_values(by=["Accuracy", "F1 Score"], ascending=False)


# ## Observation
# 
# - Even Sector wise analysis does not give us good accuracy. 

# ## Train Test Split

# In[ ]:


my_cv = 10


# In[ ]:


X = df.drop('Class', axis = 1)
y = df[['Class']]

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.75, test_size = 0.25, random_state = 42)

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


# In[ ]:


y_train_reshape = pd.DataFrame(y_train.values.reshape(-1,1))
print("Counts of label '1': {}".format((y_train_reshape==1).sum()[0]))
print("Counts of label '0': {} \n".format((y_train_reshape==0).sum()[0]))

y_train_1 = (y_train_reshape==1).sum()[0]
print("Percentage of Profitable Company : {}% \n".format(round(y_train_1/len(y_train_reshape)*100,2)))


# # Principal Component Analysis

# In[ ]:


#Importing the PCA module

pca = PCA(random_state=42)
pca_again = PCA(0.95)


# ## MinMax Scaler

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.8, test_size = 0.2, random_state = 123)


# In[ ]:


scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[ ]:


X_train_pca_mm = pca_again.fit_transform(X_train)


# In[ ]:


X_train_pca_mm.shape


# In[ ]:


# Tranforming X_Test
X_test_pca_mm = pca_again.transform(X_test)
X_test_pca_mm.shape


# In[ ]:


#Doing the PCA on the train data
pca.fit(X_train)


# In[ ]:


#Making the screeplot - plotting the cumulative variance against the number of components
get_ipython().run_line_magic('matplotlib', 'inline')
fig = plt.figure(figsize = (12,8))
plt.vlines(x = X_train_pca_mm.shape[1],ymax = 1,ymin = 0,colors = 'r',linestyles = '--')
plt.hlines(y = 0.95, xmax = 70,xmin = 0,colors = 'g',linestyles = '--')
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')

sns.despine()
plt.show()


# In[ ]:





# In[ ]:


pca = PCA(n_components=5)


# In[ ]:


scaler = MinMaxScaler()


# In[ ]:


scaled_df=df.copy()
scaled_df=pd.DataFrame(scaler.fit_transform(scaled_df), columns=scaled_df.columns)
pca_fit = pca.fit(scaled_df)


# In[ ]:





# In[ ]:


fig = plt.figure(figsize=(14,5))
PC_values = np.arange(pca_fit.n_components_) + 1

plt.plot(PC_values, pca_fit.explained_variance_ratio_, 'o-', linewidth=2, color='blue')
plt.title('Scree Plot')
plt.xlabel('Principal Component')
plt.ylabel('Variance Explained')

plt.show()


# In[ ]:


print(pca_fit.explained_variance_ratio_)


# ## StandardScaler

# In[ ]:


#Importing the PCA module

pca = PCA(random_state=42)
pca_again = PCA(0.95)


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.8, test_size = 0.2, random_state = 123)


# In[ ]:


scaler = StandardScaler()


X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[ ]:


X_train_pca_std = pca_again.fit_transform(X_train)


# In[ ]:


X_train_pca_std.shape


# In[ ]:


# Tranforming X_Test
X_test_pca_std = pca_again.transform(X_test)
X_test_pca_std.shape


# In[ ]:


#Doing the PCA on the train data
pca.fit(X_train)


# In[ ]:


#Making the screeplot - plotting the cumulative variance against the number of components
get_ipython().run_line_magic('matplotlib', 'inline')
fig = plt.figure(figsize = (12,8))
plt.vlines(x = X_train_pca_std.shape[1],ymax = 1,ymin = 0,colors = 'r',linestyles = '--')
plt.hlines(y = 0.95, xmax = 180,xmin = 0,colors = 'g',linestyles = '--')
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')

sns.despine()
plt.show()


# In[ ]:





# In[ ]:


pca = PCA(n_components=5)


# In[ ]:


scaler = StandardScaler()


# In[ ]:


scaled_df=df.copy()
scaled_df=pd.DataFrame(scaler.fit_transform(scaled_df), columns=scaled_df.columns)
pca_fit = pca.fit(scaled_df)


# In[ ]:





# In[ ]:


fig = plt.figure(figsize=(14,5))
PC_values = np.arange(pca_fit.n_components_) + 1

plt.plot(PC_values, pca_fit.explained_variance_ratio_, 'o-', linewidth=2, color='blue')
plt.title('Scree Plot')
plt.xlabel('Principal Component')
plt.ylabel('Variance Explained')

plt.show()


# In[ ]:


print(pca_fit.explained_variance_ratio_)


# # Defining functions for Modelling

# In[ ]:


# Defining the function to plot the ROC Curve

def draw_roc (actual, probs):
    fpr, tpr, thresholds = metrics.roc_curve(actual, probs, drop_intermediate = False)
    
    auc_score = metrics.roc_auc_score(actual, probs)
    plt.figure(figsize=(7,5))
    plt.plot(fpr, tpr, label="ROC curve (area = %0.2f)"%auc_score)
    
    plt.plot([0,1],[0,1],'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate or [1 - True Negative Rate]")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver operating charactersitc example")
    plt.legend(loc="lower right")
    sns.despine()
    plt.tight_layout()
    plt.show()
    
    return fpr, tpr, thresholds


# In[ ]:


metricsdataframe=pd.DataFrame(columns=['Model',
                                       'Train/Test',
                                       'Roc_auc_score',
                                       'Sensitivity',
                                       'Specificity',
                                      'FPR',
                                      'Positive predictive value',
                                      'Negative Predictive value',
                                      'Precision',
                                      'Accuracy',
                                      'F1-Score'])


# In[ ]:


# Defining function to get the evaluation metrics of the models.
def getModelMetrics(actual_Class=False,pred_Class=False,model_name='',train_or_test=''):

    confusion = metrics.confusion_matrix(actual_Class, pred_Class)

    TP = confusion[1,1] # true positive 
    TN = confusion[0,0] # true negatives
    FP = confusion[0,1] # false positives
    FN = confusion[1,0] # false negatives
    

    
    Roc_auc_score=round(metrics.roc_auc_score(actual_Class,pred_Class),2)
    # Let's see the sensitivity of our logistic regression model
    Sensitivity=round((TP / float(TP+FN)),2)
    # Let us calculate specificity
    Specificity=round((TN / float(TN+FP)),2)
    # Calculate false postive rate - predicting profit when customer does not have profitted
    FPR=round((FP/ float(TN+FP)),2)
    # positive predictive value 
    PositivePredictiveValue=round((TP / float(TP+FP)),2)
    # Negative predictive value
    NegativePredictiveValue=round((TN / float(TN+ FN)),2)
    # sklearn precision score value 
    Precision=round(metrics.precision_score(actual_Class, pred_Class ),2)
    # Accuracy
    Accuracy = round(metrics.accuracy_score(actual_Class, pred_Class), 2)
    # F-1 Score
    F1_Score = round(metrics.f1_score(actual_Class, pred_Class), 2)
    
    
    print("Roc_auc_score : {}".format(metrics.roc_auc_score(actual_Class,pred_Class)))
    # Let's see the sensitivity of our logistic regression model
    print('Sensitivity/Recall : {}'.format(TP / float(TP+FN)))
    # Let us calculate specificity
    print('Specificity: {}'.format(TN / float(TN+FP)))
    # Calculate false postive rate - predicting profit when customer does not have profitted
    print('False Positive Rate: {}'.format(FP/ float(TN+FP)))
    # positive predictive value 
    print('Positive predictive value: {}'.format(TP / float(TP+FP)))
    # Negative predictive value
    print('Negative Predictive value: {}'.format(TN / float(TN+ FN)))
    # sklearn precision score value 
    print('Precision: {}'.format(metrics.precision_score(actual_Class, pred_Class )))
    # sklearn precision score value 
    print('Accuracy: {}'.format(metrics.accuracy_score(actual_Class, pred_Class )))
    #F1 Score
    print("F1 Score: {}".format(metrics.f1_score(actual_Class, pred_Class )))
    
#     data_list=[model_name,train_or_test,Roc_auc_score,Sensitivity,Specificity,NegativePredictiveValue,Precision, ]
    data_list=[model_name,train_or_test,Roc_auc_score,Sensitivity,Specificity,FPR,PositivePredictiveValue,NegativePredictiveValue,Precision, Accuracy, F1_Score]
    series_metrics=pd.Series(data_list,index=metricsdataframe.columns)
    
    return series_metrics
    


# In[ ]:


# Defining function to get the evaluation metrics of the models.
def getModelMetrics2(actual_Class=False,pred_Class=False,model_name='',train_or_test=''):

    confusion = metrics.confusion_matrix(actual_Class, pred_Class)

    TP = confusion[1,1] # true positive 
    TN = confusion[0,0] # true negatives
    FP = confusion[0,1] # false positives
    FN = confusion[1,0] # false negatives
    

    
    Roc_auc_score=round(metrics.roc_auc_score(actual_Class,pred_Class),2)
    # Let's see the sensitivity of our logistic regression model
    Sensitivity=round((TP / float(TP+FN)),2)
    # Let us calculate specificity
    Specificity=round((TN / float(TN+FP)),2)
    # Calculate false postive rate - predicting profit when customer does not have profitted
    FPR=round((FP/ float(TN+FP)),2)
    # positive predictive value 
    PositivePredictiveValue=round((TP / float(TP+FP)),2)
    # Negative predictive value
    NegativePredictiveValue=round((TN / float(TN+ FN)),2)
    # sklearn precision score value 
    Precision=round(metrics.precision_score(actual_Class, pred_Class ),2)
    # Accuracy
    Accuracy = round(metrics.accuracy_score(actual_Class, pred_Class), 2)
    # F-1 Score
    F1_Score = round(metrics.f1_score(actual_Class, pred_Class), 2)
    

#     data_list=[model_name,train_or_test,Roc_auc_score,Sensitivity,Specificity,NegativePredictiveValue,Precision, ]
    data_list=[model_name,train_or_test,Roc_auc_score,Sensitivity,Specificity,FPR,PositivePredictiveValue,NegativePredictiveValue,Precision, Accuracy, F1_Score]
    series_metrics=pd.Series(data_list,index=metricsdataframe.columns)
    
    return series_metrics
    


# In[ ]:


# Defining function to write the evaluation metrics of the models into data frame.
def WriteModelMetrics(series_metrics,metricsdataframe):
    metricsdataframe = metricsdataframe.append(series_metrics,ignore_index=True)
    return metricsdataframe


# In[ ]:


# Function to find the optimal cutoff for classifing as Profit/non-profit
def findOptimalCutoff(df):
    
    # Let's create columns with different probability cutoffs 
    numbers = [float(x)/10 for x in range(10)]
    for i in numbers:
        df[i] = df.Class_Prob.map( lambda x: 1 if x > i else 0)
        
    
    # Now let's calculate accuracy sensitivity and specificity for various probability cutoffs.
    cutoff_df = pd.DataFrame( columns = ['prob','accuracy','sensi','speci'])
    from sklearn.metrics import confusion_matrix
    
    num = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]

    for i in num:
        cm1 = metrics.confusion_matrix(df.Class, df[i] )
        total1=sum(sum(cm1))
        accuracy = (cm1[0,0]+cm1[1,1])/total1
        
        speci = cm1[0,0]/(cm1[0,0]+cm1[0,1])
        sensi = cm1[1,1]/(cm1[1,0]+cm1[1,1])
        cutoff_df.loc[i] =[ i ,accuracy,sensi,speci]
    print(cutoff_df)
    
    # Let's plot accuracy sensitivity and specificity for various probabilities.
    cutoff_df.plot.line(x='prob', y=['accuracy','sensi','speci'])
    plt.figure(figsize=(7,5))
    plt.tight_layout()
    sns.despine()
    plt.show()


# In[ ]:


# Calculating VIF 
from statsmodels.stats.outliers_influence import variance_inflation_factor
# Defining a function to give VIF value 
def vif_cal(X):     
    vif = pd.DataFrame() 
    vif['Features'] = X.columns
    vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]    
    vif['VIF'] = round(vif['VIF'], 2)
    vif = vif.sort_values(by="VIF", ascending=False)
    return vif


# In[ ]:


# This method will result in the calculation of predicted value of the Class column.
def predictClassWithProb(model,X,y,prob,model_name='',train_or_test=''):
    
    # predict
    pred_probs = model.predict_proba(X)[:,1]
    
    y_df= pd.DataFrame({'Class':y.Class, 'Class_Prob':pred_probs})
    # Creating new column 'predicted' with 1 if Class_Prob>0.5 else 0
    y_df['final_predicted'] = y_df.Class_Prob.map( lambda x: 1 if x > prob else 0)
    # Let's see the head
    series_metrics=getModelMetrics(y_df.Class,y_df.final_predicted,model_name,train_or_test)
    return y_df,series_metrics


# In[ ]:


# This method will result in the calculation of predicted value of the Class column.
def predictClassWithProb2(model,X,y,prob,model_name='',train_or_test=''):
    
    # predict
    pred_probs = model.predict_proba(X)[:,1]
    
    y_df= pd.DataFrame({'Class':y.Class, 'Class_Prob':pred_probs})
    # Creating new column 'predicted' with 1 if Class_Prob>0.5 else 0
    y_df['final_predicted'] = y_df.Class_Prob.map( lambda x: 1 if x > prob else 0)
    # Let's see the head
    series_metrics=getModelMetrics2(y_df.Class,y_df.final_predicted,model_name,train_or_test)
    return y_df,series_metrics


# In[ ]:


# This method will perform cross-validation and the display the model report.
def modelfit(alg, X_train, y_train, performCV=True, cv_folds=my_cv):
    #Fit the algorithm on the data
    alg.fit(X_train, y_train)
        
    #Predict training set:
    dtrain_predictions = alg.predict(X_train)
    dtrain_predprob = alg.predict_proba(X_train)[:,1]
    
    #Perform cross-validation:
    if performCV:
        cv_score = cross_val_score(alg, X_train, y_train, cv=cv_folds, scoring='precision')
    
    #Print model report:
    print ("\nModel Report")
    print ("Accuracy : %.3g" % metrics.roc_auc_score(y_train, dtrain_predictions))
    print ("Recall/Sensitivity : %.3g" % metrics.recall_score(y_train, dtrain_predictions))
    print ("AUC Score (Train): %f" % metrics.roc_auc_score(y_train, dtrain_predprob))
    
    if performCV:
        print ("CV Score : Mean - %.3g | Std - %.3g | Min - %.3g | Max - %.3g" % (np.mean(cv_score),np.std(cv_score),np.min(cv_score),np.max(cv_score)))
        


# In[ ]:


# This method will plot accuracy of the model with the given param of model.
def plot_traintestAcc(score,param):
    scores = score
    # plotting accuracies with max_depth
    plt.figure()
    plt.plot(scores["param_"+param], 
    scores["mean_train_score"], 
    label="training accuracy")
    plt.plot(scores["param_"+param], 
    scores["mean_test_score"], 
    label="test accuracy")
    plt.xlabel(param)
    plt.ylabel("accuracy")
    plt.legend()
    plt.show()


# In[ ]:


# This method will plot accuracy of the random forest model.
def random_forst_feature_graph(grid_search, param):
    scores = grid_search.cv_results_
    plt.figure(figsize=(8,8))
    
    param = "param_"+param
    plt.plot(scores[param], 
             scores["mean_train_score"], 
             label="Training accuracy")

    plt.plot(scores[param], 
             scores["mean_test_score"], 
             label="Test accuracy")

    plt.xlabel(param)
    plt.ylabel("F1")
    plt.legend()

    plt.tight_layout()
    sns.despine()
    plt.show()


# # Modelling

# In[ ]:


modelData = pd.DataFrame(columns=["Model","Parameter","Value","Cut-Off"])


# In[ ]:


X_train = X_train_pca_std
X_test = X_test_pca_std


# In[ ]:


X_train_pca = X_train
y_train_res = y_train
X_test_pca = X_test
# y_test = y_test

y_train_res=y_train_res.values.reshape(-1,1)
# y_test = y_test.values.reshape(-1,1)


# In[ ]:


X_train_pca.shape


# In[ ]:


y_train_res.shape


# In[ ]:


X_test_pca.shape


# In[ ]:


y_test.shape


# In[ ]:


y_train_res = pd.DataFrame(y_train_res)


# In[ ]:


y_train_res.rename(columns={0:"Class"}, inplace=True)


# # 1. Logistic Regression

# In[ ]:


# Creating Train-Test variables for Logistic Regression
X_train_lr = pd.DataFrame(X_train_pca)
y_train_lr = pd.DataFrame(y_train_res)
X_test_lr = pd.DataFrame(X_test_pca)
y_test_lr = y_test


# In[ ]:


logml = sm.GLM(y_train_lr, (sm.add_constant(X_train_lr)), family = sm.families.Binomial())
logml.fit().summary()


# In[ ]:


# Checking for the VIF of the train data.
vif = vif_cal(X_train_lr) 
vif


# ### Observation
# - There are no Columns which are highly correlated ---> VIF = 1
# - There are not many columns whose coefficients are not statistically significant ----> p>0.05

# In[ ]:


lg = LogisticRegression()


# In[ ]:


modelfit(lg, X_train_lr, y_train_lr)


# In[ ]:


# predictions on Test data
pred_probs_test = lg.predict(X_test_lr)
getModelMetrics(y_test_lr,pred_probs_test)


# In[ ]:


print("Accuracy : {}".format(metrics.accuracy_score(y_test_lr,pred_probs_test)))
print("Recall : {}".format(metrics.recall_score(y_test_lr,pred_probs_test)))
print("Precision : {}".format(metrics.precision_score(y_test_lr,pred_probs_test)))


# In[ ]:


print(metrics.confusion_matrix(y_test_lr,pred_probs_test))


# In[ ]:


# #Making prediction on the test data
# pred_probs_test = lg.predict_proba(X_test_lr)[:,1]

# print("roc_auc_score(Test) {:2.2}".format(metrics.roc_auc_score(y_test_lr, pred_probs_test)))


# In[ ]:


HighestAOC = 0.0
cutOffAOC = 0.0

for i in np.arange(0,1,0.1):
    tenp1, tenp2 = predictClassWithProb2(lg,X_test_lr,y_test_lr,i,model_name='Logistic Regression',train_or_test='TEST')
    cutOffAOC=tenp2['Roc_auc_score']
    if (cutOffAOC>HighestAOC):
        HighestAOC = cutOffAOC
        cutOffAOC=i


# In[ ]:


y_train_lr.rename(columns={0:"Class"}, inplace=True)

cut_off_prob=cutOffAOC
temp_aoc,series_metrics = predictClassWithProb(lg,X_test_lr,y_test_lr,cut_off_prob)


# In[ ]:


draw_roc(temp_aoc.Class, temp_aoc.final_predicted)


# In[ ]:


# finding cut-off with the right balance of the metrices
# sensitivity vs specificity trade-off
findOptimalCutoff(temp_aoc)


# In[ ]:


HighestPrec = 0.0
cutOffPrec = 0.0

for i in np.arange(0.3, 0.8,0.01):
    tenp1, tenp2 = predictClassWithProb2(lg,X_test_lr,y_test_lr,i,model_name='Logistic Regression',train_or_test='TEST')
    curPrecision=tenp2['Precision']
    if (curPrecision>HighestPrec):
        HighestPrec = curPrecision
        cutOffPrec=i
        

HighestAccuracy = 0.0
cutOffAccuracy = 0.0

for i in np.arange(0, 1,0.01):
    tenp1, tenp2 = predictClassWithProb2(lg,X_test_lr,y_test_lr,i,model_name='Logistic Regression',train_or_test='TEST')
    curAccuracy=tenp2['Accuracy']
    if (curAccuracy>HighestAccuracy):
        HighestAccuracy = curAccuracy
        cutOffAccuracy=i


# In[ ]:


print("Highest Precision: ",HighestPrec, " at cut-off: ",cutOffPrec)
print("Highest Accuracy: ",HighestAccuracy, " at cut-off: ",cutOffAccuracy)


# In[ ]:


### predicting with the choosen cut-off on TEST
cut_off_prob=cutOffAccuracy
x_temp,series_metrics = predictClassWithProb(lg,X_test_lr,y_test_lr,cut_off_prob,model_name='Logistic Regression',train_or_test='TEST')
metricsdataframe=WriteModelMetrics(series_metrics,metricsdataframe)


# In[ ]:


### predicting with the choosen cut-off on TEST
cut_off_prob=cutOffPrec
x_temp,series_metrics = predictClassWithProb(lg,X_test_lr,y_test_lr,cut_off_prob,model_name='Logistic Regression',train_or_test='TEST')
metricsdataframe=WriteModelMetrics(series_metrics,metricsdataframe)


# In[ ]:


modelData = modelData.append({"Model":"Logistic Regression", 
                              "Parameter": "Precision","Value": HighestPrec,
                              "Cut-Off":cutOffPrec}, ignore_index=True)

modelData = modelData.append({"Model": "Logistic Regression", 
                              "Parameter": "Accuracy","Value": HighestAccuracy,
                              "Cut-Off":cutOffAccuracy}, ignore_index=True)


# In[ ]:


modelData


# # 2. Decision Tree

# In[ ]:


# Creating Train-Test variables for Decision Tree
X_train_dt = pd.DataFrame(X_train_pca)
y_train_dt = pd.DataFrame(y_train_res)
X_test_dt = pd.DataFrame(X_test_pca)
y_test_dt = y_test


# In[ ]:


X_train_dt.shape


# In[ ]:


y_train_dt.shape


# In[ ]:


X_test_dt.shape, y_test_dt.shape


# In[ ]:


##### Applying Decision Tree Classifier on our principal components with Hyperparameter tuning
dt = DecisionTreeClassifier(class_weight='balanced',
                             max_features='auto',
                             min_samples_split=100,
                             min_samples_leaf=100,
                             max_depth=10,
                             random_state=123)

modelfit(dt, X_train_dt, y_train_dt)


# In[ ]:


# Create the parameter grid based on the results of random search 
param_grid = {
    'max_depth': [5,10,15,20,30,50],
    'min_samples_leaf': range(100, 500, 50),
    'min_samples_split': range(100, 500, 50),
    'max_features': [5,10,15,20,30,50]
}
# Create a base model
dt = DecisionTreeClassifier(class_weight='balanced',random_state=123)

# Instantiate the grid search model
grid_search = GridSearchCV(estimator = dt, param_grid = param_grid, 
                           cv = my_cv, n_jobs = -1,verbose = 1000,scoring="precision")


# In[ ]:


# Fit the grid search to the data
grid_search.fit(X_train_dt, y_train_dt)


# In[ ]:


# printing the optimal accuracy score and hyperparameters
print('We can get score of',grid_search.best_score_,'using',grid_search.best_params_)


# In[ ]:


cv_df = pd.DataFrame(grid_search.cv_results_)
cv_df.head(3)


# In[ ]:


cv_df.nlargest(3,"mean_test_score")


# In[ ]:


grid_search.best_score_


# In[ ]:


grid_search.best_estimator_


# In[ ]:


param_max_depth = cv_df.nlargest(3,"mean_test_score").param_max_depth.iloc[0]
param_max_features = cv_df.nlargest(3,"mean_test_score").param_max_features.iloc[0]
param_min_samples_leaf = cv_df.nlargest(3,"mean_test_score").param_min_samples_leaf.iloc[0]
param_min_samples_split = cv_df.nlargest(3,"mean_test_score").param_min_samples_split.iloc[0]


# In[ ]:


# model with the best hyperparameters
dt_final = DecisionTreeClassifier(class_weight='balanced',
                             max_depth=param_max_depth,
                             max_features=param_max_features,
                             min_samples_leaf=param_min_samples_leaf, 
                             min_samples_split=param_min_samples_split,
                             random_state=123)


# In[ ]:


modelfit(dt_final,X_train_dt,y_train_dt)


# In[ ]:


HighestAOC = 0.0
cutOffAOC = 0.0

for i in np.arange(0,1,0.1):
    tenp1, tenp2 = predictClassWithProb2(dt_final,X_test_dt,y_test_dt,i,model_name='Decision tree',train_or_test='TEST')
    cutOffAOC=tenp2['Roc_auc_score']
    if (cutOffAOC>HighestAOC):
        HighestAOC = cutOffAOC
        cutOffAOC=i


# In[ ]:


y_train_dt.rename(columns={0:"Class"}, inplace=True)

cut_off_prob=cutOffAOC
temp_aoc,series_metrics = predictClassWithProb(dt_final,X_train_dt,y_train_dt,cut_off_prob)


# In[ ]:


draw_roc(temp_aoc.Class, temp_aoc.final_predicted)
print("roc_auc_score : {:2.2f}".format(metrics.roc_auc_score(temp_aoc.Class, temp_aoc.final_predicted)))


# In[ ]:


# make predictions
pred_probs_test = dt_final.predict(X_test_dt)
#Let's check the model metrices.
getModelMetrics(actual_Class=y_test_dt,pred_Class=pred_probs_test)


# In[ ]:


# finding cut-off with the right balance of the metrices
# sensitivity vs specificity trade-off
findOptimalCutoff(temp_aoc)


# In[ ]:


dt_final


# In[ ]:


HighestPrec = 0.0
cutOffPrec = 0.0

for i in np.arange(0.3, 0.8,0.01):
    tenp1, tenp2 = predictClassWithProb2(dt_final,X_test_dt,y_test_dt,i,model_name='Decision Tree',train_or_test='TEST')
    curPrecision=tenp2['Precision']
    if (curPrecision>HighestPrec):
        HighestPrec = curPrecision
        cutOffPrec=i
        

HighestAccuracy = 0.0
cutOffAccuracy = 0.0

for i in np.arange(0.02, 1,0.01):
    tenp1, tenp2 = predictClassWithProb2(dt_final,X_test_dt,y_test_dt,i,model_name='Decision Tree',train_or_test='TEST')
    curAccuracy=tenp2['Accuracy']
    if (curAccuracy>HighestAccuracy):
        HighestAccuracy = curAccuracy
        cutOffAccuracy=i


# In[ ]:


print("Highest Precision: ",HighestPrec, " at cut-off: ",cutOffPrec)
print("Highest Accuracy: ",HighestAccuracy, " at cut-off: ",cutOffAccuracy)


# In[ ]:





# In[ ]:


cut_off_prob = cutOffPrec
y_train_df,series_metrics = predictClassWithProb(dt_final,X_train_dt,y_train_dt,cut_off_prob)


# In[ ]:


cut_off_prob = cutOffAccuracy
y_train_df,series_metrics = predictClassWithProb(dt_final,X_train_dt,y_train_dt,cut_off_prob)


# In[ ]:


modelData = modelData.append({"Model":"Decision Tree", 
                              "Parameter": "Precision","Value": HighestPrec,
                              "Cut-Off":cutOffPrec}, ignore_index=True)

modelData = modelData.append({"Model": "Decision Tree", 
                              "Parameter": "Accuracy","Value": HighestAccuracy,
                              "Cut-Off":cutOffAccuracy}, ignore_index=True)


# In[ ]:


modelData


# # 3. Random Forest

# In[ ]:


# Creating Train-Test variables for Random Forest
X_train_rf = pd.DataFrame(X_train_pca)
y_train_rf = pd.DataFrame(y_train_res)
X_test_rf = pd.DataFrame(X_test_pca)
y_test_rf = y_test


# In[ ]:


rf = RandomForestClassifier(random_state=42, max_depth=5, n_estimators=20, oob_score=True)


# In[ ]:


rf.fit(X_train_rf, y_train_rf)


# In[ ]:


rf.oob_score_


# In[ ]:


# make predictions
pred_probs_test = rf.predict(X_test_rf)

#Let's check the model metrices.
getModelMetrics(actual_Class=y_test_rf,pred_Class=pred_probs_test)


# In[ ]:


parameters = {'max_depth': range(5, 40, 5)}
rf = RandomForestClassifier()
grid_search = GridSearchCV(rf, parameters, cv=my_cv, scoring="precision", verbose=1000, return_train_score=True)

grid_search.fit(X_train_rf, y_train_rf)


# In[ ]:


pd.DataFrame(grid_search.cv_results_).sort_values(by="rank_test_score")


# In[ ]:


# grid_search.cv_results_

plot_traintestAcc(grid_search.cv_results_,'max_depth')


# In[ ]:


my_max_depth = pd.DataFrame(grid_search.cv_results_).sort_values(by="rank_test_score")['param_max_depth'].iloc[0]


# ### Tuning n_estimators
# 

# In[ ]:


parameters = {'n_estimators': range(5, 70, 5)}

rf = RandomForestClassifier(max_depth=my_max_depth,random_state=10)
grid_search = GridSearchCV(rf, parameters, cv=my_cv, scoring="precision", verbose=100, return_train_score=True)


grid_search.fit(X_train_rf, y_train_rf)


# In[ ]:


pd.DataFrame(grid_search.cv_results_).sort_values(by="rank_test_score").head()


# In[ ]:


random_forst_feature_graph(grid_search, "n_estimators")


# In[ ]:


my_n_estimator = pd.DataFrame(grid_search.cv_results_).sort_values(by="rank_test_score")['param_n_estimators'].iloc[0]


# ### Tuning max_features

# In[ ]:


parameters = {'max_features': [5, 10, 15, 20, 25, 30,50,70]}

rf = RandomForestClassifier(max_depth=my_max_depth,n_estimators = my_n_estimator,random_state=10)
grid_search = GridSearchCV(rf, parameters, cv=my_cv, scoring="precision", verbose=1000, return_train_score=True)


grid_search.fit(X_train_rf, y_train_rf)


# In[ ]:


pd.DataFrame(grid_search.cv_results_).sort_values(by="rank_test_score")


# In[ ]:


random_forst_feature_graph(grid_search, "max_features")


# In[ ]:


my_max_features=pd.DataFrame(grid_search.cv_results_).sort_values(by="rank_test_score")['param_max_features'].iloc[0]


# ### Tuning min_samples_leaf

# In[ ]:


parameters = {'min_samples_leaf': range(1, 500, 50)}

rf = RandomForestClassifier(max_depth=my_max_depth,n_estimators = my_n_estimator, max_features = my_max_features, random_state=10)
grid_search = GridSearchCV(rf, parameters, cv=my_cv, scoring="precision", verbose=1000, return_train_score=True)

grid_search.fit(X_train_rf, y_train_rf)


# In[ ]:


pd.DataFrame(grid_search.cv_results_).sort_values(by="rank_test_score")


# In[ ]:


random_forst_feature_graph(grid_search, "min_samples_leaf")


# In[ ]:


my_min_sample_leaf = pd.DataFrame(grid_search.cv_results_).sort_values(by="rank_test_score")['param_min_samples_leaf'].iloc[0]


# ### Tuning min_samples_split

# In[ ]:


parameters = {'min_samples_split': range(50, 550, 50)}

rf = RandomForestClassifier(max_depth=my_max_depth,n_estimators = my_n_estimator, max_features = my_max_features, min_samples_leaf = my_min_sample_leaf,random_state=10)
grid_search = GridSearchCV(rf, parameters, cv=my_cv, scoring="precision", verbose=1000, return_train_score=True)


grid_search.fit(X_train_rf, y_train_rf)


# In[ ]:


pd.DataFrame(grid_search.cv_results_).sort_values(by="rank_test_score")


# In[ ]:


random_forst_feature_graph(grid_search, "min_samples_split")


# In[ ]:


my_min_samples_split=pd.DataFrame(grid_search.cv_results_).sort_values(by="rank_test_score")['param_min_samples_split'].iloc[0]


# In[ ]:


### Tuning min_samples_leaf

parameters = {'min_samples_leaf': range(1, 500, 50)}

rf = RandomForestClassifier(max_depth=my_max_depth,n_estimators = my_n_estimator, max_features = my_max_features, random_state=10)
grid_search = GridSearchCV(rf, parameters, cv=my_cv, scoring="precision", verbose=1000, return_train_score=True)

grid_search.fit(X_train_rf, y_train_rf)

pd.DataFrame(grid_search.cv_results_).sort_values(by="rank_test_score")

random_forst_feature_graph(grid_search, "min_samples_leaf")

my_min_sample_leaf = pd.DataFrame(grid_search.cv_results_).sort_values(by="rank_test_score")['param_min_samples_leaf'].iloc[0]

### Tuning min_samples_split

parameters = {'min_samples_split': range(50, 550, 50)}

rf = RandomForestClassifier(max_depth=my_max_depth,n_estimators = my_n_estimator, max_features = my_max_features, min_samples_leaf = my_min_sample_leaf,random_state=10)
grid_search = GridSearchCV(rf, parameters, cv=my_cv, scoring="precision", verbose=1000, return_train_score=True)


grid_search.fit(X_train_rf, y_train_rf)

pd.DataFrame(grid_search.cv_results_).sort_values(by="rank_test_score")

random_forst_feature_graph(grid_search, "min_samples_split")

my_min_samples_split=pd.DataFrame(grid_search.cv_results_).sort_values(by="rank_test_score")['param_min_samples_split'].iloc[0]


# ### Final Model

# In[ ]:


rf_final = RandomForestClassifier(max_depth=my_max_depth,
                                  n_estimators = my_n_estimator, 
                                  max_features = my_max_features, 
                                  min_samples_leaf = my_min_sample_leaf,
                                  min_samples_split=my_min_samples_split,
                                  random_state=123)


# In[ ]:


print("Model performance on Train data:")
modelfit(rf_final,X_train_rf,y_train_rf)


# In[ ]:


# predict on test data
predictions = rf_final.predict(X_test_rf)


# In[ ]:


print("Model performance on Test data:")
getModelMetrics(y_test_rf,predictions)


# In[ ]:


# predicting with default cut-off 0.5
cut_off_prob=0.5
y_train_df,series_metrics = predictClassWithProb(rf_final,X_train_rf,y_train_rf,cut_off_prob)


# In[ ]:


# finding cut-off with the right balance of the metrices
findOptimalCutoff(y_train_df)


# In[ ]:


HighestPrec = 0.0
cutOffPrec = 0.0

for i in np.arange(0.3, 0.7,0.01):
    tenp1, tenp2 = predictClassWithProb2(rf_final,X_test_rf,y_test_rf,i,model_name='Random Forest',train_or_test='TEST')
    curPrecision=tenp2['Precision']
    if (curPrecision>HighestPrec):
        HighestPrec = curPrecision
        cutOffPrec=i
        

HighestAccuracy = 0.0
cutOffAccuracy = 0.0

for i in np.arange(0.02, 1,0.01):
    tenp1, tenp2 = predictClassWithProb2(rf_final,X_test_rf,y_test_rf,i,model_name='Random Forest',train_or_test='TEST')
    curAccuracy=tenp2['Accuracy']
    if (curAccuracy>HighestAccuracy):
        HighestAccuracy = curAccuracy
        cutOffAccuracy=i


# In[ ]:


print("Highest Precision: ",HighestPrec, " at cut-off: ",cutOffPrec)
print("Highest Accuracy: ",HighestAccuracy, " at cut-off: ",cutOffAccuracy)


# In[ ]:


# Testing Data
cut_off_prob=cutOffPrec
y_test_df,series_metrics= predictClassWithProb(rf_final,X_test_rf,y_test_rf,cut_off_prob,model_name='Random Forest',train_or_test='TEST')
metricsdataframe=WriteModelMetrics(series_metrics,metricsdataframe)


# In[ ]:


# Testing Data
cut_off_prob=cutOffAccuracy
y_test_df,series_metrics= predictClassWithProb(rf_final,X_test_rf,y_test_rf,cut_off_prob,model_name='Random Forest',train_or_test='TEST')
metricsdataframe=WriteModelMetrics(series_metrics,metricsdataframe)


# In[ ]:


modelData = modelData.append({"Model":"Random Forest", 
                              "Parameter": "Precision","Value": HighestPrec,
                              "Cut-Off":cutOffPrec}, ignore_index=True)

modelData = modelData.append({"Model": "Random Forest", 
                              "Parameter": "Accuracy","Value": HighestAccuracy,
                              "Cut-Off":cutOffAccuracy}, ignore_index=True)


# In[ ]:


modelData


# # 4. Gradient Boosting

# In[ ]:


# Creating Train-Test variables for Gradient Boosting
X_train_gb = pd.DataFrame(X_train_pca)
y_train_gb = pd.DataFrame(y_train_res)
X_test_gb = pd.DataFrame(X_test_pca)
y_test_gb = y_test


# In[ ]:


from sklearn.ensemble import GradientBoostingClassifier  #GBM algorithm

# Fitting the default GradientBoostingClassifier
gbm = GradientBoostingClassifier(random_state=10)
modelfit(gbm, X_train_gb, y_train_gb)


# In[ ]:


param = {'n_estimators':range(10,170,20), 
         'max_depth':range(4,18,2), 
         'min_samples_split':range(250,801,250), 
        'max_features':range(5,550,50)}


gbm = GradientBoostingClassifier(random_state=10)
grid_search = GridSearchCV(estimator = gbm, param_grid = param, scoring='precision',n_jobs=-1,verbose=1000, cv=3)


grid_search.fit(X_train_gb, y_train_gb.values.ravel())


# In[ ]:


grid_search.best_params_


# In[ ]:


grid_search.best_score_


# In[ ]:


pd.DataFrame(grid_search.cv_results_).sort_values(by="rank_test_score")


# In[ ]:


md_gb = pd.DataFrame(grid_search.cv_results_).sort_values(by="rank_test_score")['param_max_depth'].iloc[0]
param_min_samples_split_gd = pd.DataFrame(grid_search.cv_results_).sort_values(by="rank_test_score")['param_min_samples_split'].iloc[0]
param_n_estimators_gd = pd.DataFrame(grid_search.cv_results_).sort_values(by="rank_test_score")['param_n_estimators'].iloc[0]
param_max_features_gd = pd.DataFrame(grid_search.cv_results_).sort_values(by="rank_test_score")['param_max_features'].iloc[0]


# In[ ]:


# GradientBoostingClassifier with tuned Parameters
gbm_final = GradientBoostingClassifier(learning_rate=0.01, n_estimators=param_n_estimators_gd, max_features=param_max_features_gd, max_depth=md_gb, 
                                       min_samples_split=param_min_samples_split_gd, random_state=10)

modelfit(gbm_final, X_train_gb, y_train_gb)


# In[ ]:


# predictions on Test data
dtest_predictions = gbm_final.predict(X_test_gb)

# model Performance on test data
getModelMetrics(y_test_gb,dtest_predictions)


# In[ ]:


# predicting with default cut-off 0.5
cut_off_prob=0.5
y_train_df,series_metrics = predictClassWithProb(gbm_final,X_train_gb,y_train_gb,cut_off_prob)


# In[ ]:


findOptimalCutoff(y_train_df)


# In[ ]:





# In[ ]:


HighestPrec = 0.0
cutOffPrec = 0.0

for i in np.arange(0.3, 0.65,0.01):
    tenp1, tenp2 = predictClassWithProb2(gbm_final,X_test_gb,y_test_gb,i,model_name='Gradient Boosting',train_or_test='TEST')
    curPrecision=tenp2['Precision']
    if (curPrecision>HighestPrec):
        HighestPrec = curPrecision
        cutOffPrec=i
        

HighestAccuracy = 0.0
cutOffAccuracy = 0.0

for i in np.arange(0.02, 1,0.01):
    tenp1, tenp2 = predictClassWithProb2(gbm_final,X_test_gb,y_test_gb,i,model_name='Gradient Boosting',train_or_test='TEST')
    curAccuracy=tenp2['Accuracy']
    if (curAccuracy>HighestAccuracy):
        HighestAccuracy = curAccuracy
        cutOffAccuracy=i


# In[ ]:


print("Highest Precision: ",HighestPrec, " at cut-off: ",cutOffPrec)
print("Highest Accuracy: ",HighestAccuracy, " at cut-off: ",cutOffAccuracy)


# In[ ]:


# Testing Data
cut_off_prob=cutOffPrec
y_test_df,series_metrics= predictClassWithProb(gbm_final,X_test_gb,y_test_gb,cut_off_prob,model_name='Gradient Boosting',train_or_test='TEST')
metricsdataframe=WriteModelMetrics(series_metrics,metricsdataframe)


# In[ ]:


# Testing Data
cut_off_prob=cutOffAccuracy
y_test_df,series_metrics= predictClassWithProb(gbm_final,X_test_gb,y_test_gb,cut_off_prob,model_name='Gradient Boosting',train_or_test='TEST')
metricsdataframe=WriteModelMetrics(series_metrics,metricsdataframe)


# In[ ]:


modelData = modelData.append({"Model":"Gradient Boosting", 
                              "Parameter": "Precision","Value": HighestPrec,
                              "Cut-Off":cutOffPrec}, ignore_index=True)

modelData = modelData.append({"Model": "Gradient Boosting", 
                              "Parameter": "Accuracy","Value": HighestAccuracy,
                              "Cut-Off":cutOffAccuracy}, ignore_index=True)


# In[ ]:


modelData


# In[ ]:





# In[ ]:





# # 5. XG Boosting

# In[ ]:


X_train_pca = X_train_pca_std
X_test_pca = X_test_pca_std


# In[ ]:


# Creating Train-Test variables for XGBoost
X_train_xgb = pd.DataFrame(X_train_pca)
y_train_xgb = pd.DataFrame(y_train)
X_test_xgb = pd.DataFrame(X_test_pca)
y_test_xgb = y_test


# In[ ]:


# Fitting the XGBClassifier without HyperParameter Tuning
xgb = XGBClassifier(learning_rate =0.1,
                    n_estimators=1000,
                    max_depth=5,
                    min_child_weight=1,
                    gamma=0,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    nthread=-1,
                    seed=27)


# In[ ]:


# Model fit and performance on Train data
modelfit(xgb, X_train_xgb, y_train_xgb)


# In[ ]:


param_test1 = {'max_depth':range(3,10,2),'min_child_weight':range(1,6,2)}

grid_search = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=140, max_depth=5,
                                                     min_child_weight=1, gamma=0, subsample=0.8, 
                                                     colsample_bytree=0.8,
                                                     nthread=-1, scale_pos_weight=1, seed=27), 
                           param_grid = param_test1, scoring='precision',n_jobs=-1, cv=my_cv)

grid_search.fit(X_train_xgb, y_train_xgb)


# In[ ]:


grid_search.best_params_


# In[ ]:


grid_search.best_score_


# In[ ]:


my_md = grid_search.best_params_['max_depth']
my_min_child_weight = grid_search.best_params_['min_child_weight']


# In[ ]:



param_test2 = {'gamma':[i/10.0 for i in range(0,5)]}
grid_search = GridSearchCV(estimator = XGBClassifier( learning_rate=0.1, n_estimators=140, max_depth=my_md,
                                                     min_child_weight=my_min_child_weight, gamma=0, subsample=0.8, colsample_bytree=0.8, 
                                                     objective= 'binary:logistic', nthread=-1, scale_pos_weight=1,seed=27), 
                           param_grid = param_test2, scoring='precision',n_jobs=-1, cv=my_cv)

grid_search.fit(X_train_xgb, y_train_xgb)


# In[ ]:


grid_search.best_params_


# In[ ]:


my_gamma = grid_search.best_params_['gamma']


# In[ ]:


grid_search.best_score_


# In[ ]:


# Final XGBClassifier
xgb = XGBClassifier( learning_rate=0.1, n_estimators=140, max_depth=my_md,
                    min_child_weight=my_min_child_weight, gamma=my_gamma, subsample=0.8, colsample_bytree=0.8,
                    objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27)


# In[ ]:


# Fit Train data
modelfit(xgb, X_train_xgb, y_train_xgb)


# In[ ]:


# Prediction on Test data
dtest_predictions = xgb.predict(X_test_xgb)


# In[ ]:


# Model evaluation on Test data
getModelMetrics(y_test_xgb,dtest_predictions)


# In[ ]:


# predicting with default cut-off 0.5
cut_off_prob=0.5
y_train_df,series_metrics = predictClassWithProb(xgb,X_train_xgb,y_train_xgb,cut_off_prob)
y_train_df.head()


# In[ ]:


# Finding optimal cut-off probability
findOptimalCutoff(y_train_df)


# In[ ]:





# In[ ]:





# In[ ]:


HighestPrec = 0.0
cutOffPrec = 0.0

for i in np.arange(0.3, 0.7,0.01):
    tenp1, tenp2 = predictClassWithProb2(xgb,X_test_xgb,y_test_xgb,i,model_name='XG Boosting',train_or_test='TEST')
    curPrecision=tenp2['Precision']
    if (curPrecision>HighestPrec):
        HighestPrec = curPrecision
        cutOffPrec=i
        

HighestAccuracy = 0.0
cutOffAccuracy = 0.0

for i in np.arange(0.02, 1,0.01):
    tenp1, tenp2 = predictClassWithProb2(xgb,X_test_xgb,y_test_xgb,i,model_name='XG Boosting',train_or_test='TEST')
    curAccuracy=tenp2['Accuracy']
    if (curAccuracy>HighestAccuracy):
        HighestAccuracy = curAccuracy
        cutOffAccuracy=i


# In[ ]:


print("Highest Precision: ",HighestPrec, " at cut-off: ",cutOffPrec)
print("Highest Accuracy: ",HighestAccuracy, " at cut-off: ",cutOffAccuracy)


# In[ ]:


# Testing Data
cut_off_prob=cutOffPrec
y_test_df,series_metrics= predictClassWithProb(xgb,X_test_xgb,y_test_xgb,cut_off_prob,model_name='Gradient Boosting',train_or_test='TEST')
metricsdataframe=WriteModelMetrics(series_metrics,metricsdataframe)


# In[ ]:


# Testing Data
cut_off_prob=cutOffAccuracy
y_test_df,series_metrics= predictClassWithProb(xgb,X_test_xgb,y_test_xgb,cut_off_prob,model_name='Gradient Boosting',train_or_test='TEST')
metricsdataframe=WriteModelMetrics(series_metrics,metricsdataframe)


# In[ ]:


modelData = modelData.append({"Model":"XG Boosting", 
                              "Parameter": "Precision","Value": HighestPrec,
                              "Cut-Off":cutOffPrec}, ignore_index=True)

modelData = modelData.append({"Model": "XG Boosting", 
                              "Parameter": "Accuracy","Value": HighestAccuracy,
                              "Cut-Off":cutOffAccuracy}, ignore_index=True)


# In[ ]:


modelData


# In[ ]:





# In[ ]:





# # 6. Support Vector Machine

# In[ ]:


# Creating Train-Test variables for SVM
X_train_svm = pd.DataFrame(X_train_pca)
y_train_svm = pd.DataFrame(y_train_res)
X_test_svm = pd.DataFrame(X_test_pca)
y_test_svm = y_test


# In[ ]:


# instantiate an object of class SVC()
# note that we are using cost C=1
svm0 = SVC(C = 1)


# In[ ]:


# fit
svm0.fit(X_train_svm, y_train_svm)

# predict on train
y_pred = svm0.predict(X_train_svm)

series_metrics=getModelMetrics(y_train_svm,y_pred,model_name='SVM',train_or_test='TRAIN')
metricsdataframe=WriteModelMetrics(series_metrics,metricsdataframe)


# In[ ]:


# Predict on test
y_pred = svm0.predict(X_test_svm)
getModelMetrics(y_test_svm,y_pred)


# ## Hyper Parameter Tuning
# 
# ### Linear Kernal

# In[ ]:


# specify range of parameters (C) as a list
params = {"C": [0.001, 0.01, 0.1, 1, 10, 100, 500, 1000]}

svm1 = SVC()

# set up grid search scheme
# note that we are still using the 5 fold CV scheme
model_cv = GridSearchCV(estimator = svm1, param_grid = params, 
                        scoring= 'precision', cv = my_cv, verbose = 1, n_jobs=-1,
                       return_train_score=True) 

model_cv.fit(X_train_svm, y_train_svm.values.ravel())


# In[ ]:


plot_traintestAcc(model_cv.cv_results_,'C')


# In[ ]:


model_cv.best_params_


# In[ ]:


#Trying smaller values as smaller values perform better


# In[ ]:


svm_final = SVC(C = model_cv.best_params_['C'], kernel="rbf")
# fit
svm_final.fit(X_train_svm, y_train_svm.values.ravel())


# In[ ]:


# predict
y_pred = svm_final.predict(X_test_svm)


# In[ ]:


series_metrics=getModelMetrics(y_test_svm,y_pred,model_name='SVM-rbf',train_or_test='TEST')
metricsdataframe=WriteModelMetrics(series_metrics,metricsdataframe)


# ### Using Sigmoid kernal

# In[ ]:


svm_k = SVC(C = model_cv.best_params_['C'], kernel='sigmoid')
svm_k.fit(X_train_svm, y_train_svm)


# In[ ]:


y_pred = svm_k.predict(X_test_svm)


# In[ ]:


series_metrics=getModelMetrics(y_test_svm,y_pred,model_name='SVM-sigmoid',train_or_test='TEST')
metricsdataframe=WriteModelMetrics(series_metrics,metricsdataframe)


# ### SVM Poly Kernal

# In[ ]:


svm_k = SVC(C = model_cv.best_params_['C'], kernel='poly',gamma='auto')
svm_k.fit(X_train_svm, y_train_svm)


# In[ ]:


y_pred = svm_k.predict(X_test_svm)


# In[ ]:


series_metrics=getModelMetrics(y_test_svm,y_pred,model_name='SVM-poly',train_or_test='TEST')
metricsdataframe=WriteModelMetrics(series_metrics,metricsdataframe)


# In[ ]:


metricsdataframe


# # 7. Naive Bayes

# In[ ]:


# Creating Train-Test variables for SVM
X_train_nb = pd.DataFrame(X_train_pca)
y_train_nb = pd.DataFrame(y_train_res)
X_test_nb = pd.DataFrame(X_test_pca)
y_test_nb = y_test


# ## GridCV

# In[ ]:


# specify range of parameters (C) as a list
params = {"var_smoothing": [1e-1, 1e-3, 1e-5, 1e-9, 1e-12]}

nb1 = GaussianNB()

# set up grid search scheme
# note that we are still using the 5 fold CV scheme
model_cv = GridSearchCV(estimator = nb1, param_grid = params, 
                        scoring= 'precision', cv = my_cv, verbose = 1, n_jobs=-1,
                       return_train_score=True) 

model_cv.fit(X_train_nb, y_train_nb.values.ravel())


# In[ ]:


plot_traintestAcc(model_cv.cv_results_,'var_smoothing')


# In[ ]:


model_cv.best_params_


# In[ ]:


model_cv.best_params_['var_smoothing']


# In[ ]:





# In[ ]:


nb = GaussianNB(var_smoothing=model_cv.best_params_['var_smoothing'])


# In[ ]:


# fit
nb.fit(X_train_nb, y_train_nb)

# predict on train
y_pred = nb.predict(X_train_nb)

series_metrics=getModelMetrics(y_train_nb,y_pred,model_name='Naive Bayes',train_or_test='TRAIN')
metricsdataframe=WriteModelMetrics(series_metrics,metricsdataframe)


# In[ ]:


# pd.DataFrame({'col':y_pred}).value_counts()


# In[ ]:


# Predict on test
y_pred = nb.predict(X_test_nb)
getModelMetrics(y_test_nb,y_pred)


# In[ ]:





# In[ ]:




