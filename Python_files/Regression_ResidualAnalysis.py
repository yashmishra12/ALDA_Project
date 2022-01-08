#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[2]:


# !pip install lazypredict


# In[3]:


# !pip uninstall scikit-learn -y
# # 


# In[4]:


# !pip install scikit-learn==0.23.1


# ### Reading CSV files

# In[5]:


df_2014 = pd.read_csv("2014_Financial_Data.csv")
df_2015 = pd.read_csv("2015_Financial_Data.csv")
df_2016 = pd.read_csv("2016_Financial_Data.csv")
df_2017 = pd.read_csv("2017_Financial_Data.csv")
df_2018 = pd.read_csv("2018_Financial_Data.csv")


# In[6]:


nasdaq = pd.read_csv("nasdaq.csv")


# In[7]:


nasdaq.shape


# In[8]:


nasdaq.columns


# In[9]:


nasdaq.head()


# In[10]:


nasdaq.drop('Name', axis=1, inplace=True)


# In[11]:


df_2014.shape


# In[12]:


df_2015.shape


# In[13]:


df_2016.shape


# In[14]:


df_2017.shape


# In[15]:


df_2018.shape


# In[16]:


df_2014['Year'] = 2014
df_2015['Year'] = 2015
df_2016['Year'] = 2016
df_2017['Year'] = 2017
df_2018['Year'] = 2018


# In[17]:


df_2014.columns


# In[18]:


df_2014.rename(columns={'2015 PRICE VAR [%]':'Next_Year_Price_Var[%]'}, inplace=True)
df_2015.rename(columns={'2016 PRICE VAR [%]':'Next_Year_Price_Var[%]'}, inplace=True)
df_2016.rename(columns={'2017 PRICE VAR [%]':'Next_Year_Price_Var[%]'}, inplace=True)
df_2017.rename(columns={'2018 PRICE VAR [%]':'Next_Year_Price_Var[%]'}, inplace=True)
df_2018.rename(columns={'2019 PRICE VAR [%]':'Next_Year_Price_Var[%]'}, inplace=True)


# In[19]:


df = pd.concat([df_2014, df_2015, df_2016, df_2017, df_2018], axis = 0)


# In[20]:


df.shape


# In[21]:


df.columns


# In[22]:


df.head()


# In[23]:


df.rename(columns={"Unnamed: 0": "Symbol"}, inplace=True)
# df_2014.rename(columns={'2015 PRICE VAR [%]':'Next_Year_Price_Var[%]'}, inplace=True)


# In[24]:


df = pd.merge(df, nasdaq, how="inner", on="Symbol")


# In[25]:


df.shape


# In[26]:


df.head()


# In[27]:


df.Country.value_counts()


# In[28]:


df.Country.isnull().sum()


# In[ ]:





# In[29]:


#Next_Year_Price_Var[%] +ve ---> class = 1, if -ve -----> class = 0
# df.drop('Next_Year_Price_Var[%]', axis=1, inplace=True)


# In[30]:


df.rename(columns={"Symbol":"Name"}, inplace=True)


# In[31]:


df.Name.nunique()


# In[32]:


df.shape


# In[33]:


df.info


# In[34]:


df.describe()


# In[35]:


## Removing "Year" because our future companies will have different years and it should not affect our final call
df.drop("Year", axis=1, inplace=True)


# In[36]:


## Removing "Name" because our future companies will have different Name and it should not affect our final call
df.drop("Name", axis=1, inplace=True)


# In[37]:


df.head(2)


# In[38]:


df.loc[(df.Country=="United States")].shape


# In[39]:


df.loc[(df.Country=="United States")]


# In[40]:


# Selecting only United States


# In[41]:


df.Country.value_counts()


# In[42]:


df = df.loc[(df.Country=="United States")]


# In[43]:


df.drop('Country', axis=1, inplace=True)


# ### Understanding Null Value Distribution

# In[44]:


df.isnull().sum().sort_values(ascending=False)


# In[45]:


(df.isnull().sum() * 100 / len(df)).sort_values(ascending=False)


# In[46]:


(df.isnull().sum() * 100 / len(df)).sort_values(ascending=True).head(15)


# In[47]:


# all cols have some null values
len(df.isnull().any())


# In[48]:


# No columns with all null values
df.columns[df.isnull().all()]


# In[49]:


# Defining a funtion to add the count/frequency values as annotation to histogram.
def annotate_graph(ax):
    for bar in ax.patches:         
        ax.annotate(format((bar.get_height()), '.0f'),                    
                    (bar.get_x() + bar.get_width() / 2,  bar.get_height()),                    
                    ha='center', va='center',                    
                    size=10, xytext=(0, 8),                    
                    textcoords='offset points')
    return ax


# In[50]:


# Plotting histogram for the dataframe and columns having null values.
plt.figure(figsize=(28,10))

ax = sns.histplot(round((df.isnull().sum()/len(df.index) * 100).sort_values(ascending=False), 2))
ax = annotate_graph(ax)

ax.set(xticks=np.arange(0,101))
ax.set(xlabel='Null value percentage', ylabel='Count of columns with null values')
sns.despine()
plt.tight_layout()


# In[51]:


msno.matrix(df)


# In[52]:


# defining a function to get more than cutoff percent missing value

def get_missing_value_percentage(cutoff):
    y = pd.DataFrame( round((df.isnull().sum()/len(df.index) * 100).sort_values(ascending=False), 2))
    y.rename(columns={0:"Percentage"}, inplace=True)
    y2 = y[y.Percentage>cutoff]
    return y2


# In[53]:


# get columns with more than 70% missing values
greater_than_70 = get_missing_value_percentage(70)


# In[54]:


len(greater_than_70)


# In[55]:


greater_than_70


# In[56]:


# get columns with more than 50% missing values
greater_than_50 = get_missing_value_percentage(50)


# In[57]:


len(greater_than_50)


# In[58]:


greater_than_50


# In[59]:


# get columns with more than 20% missing values
greater_than_20 = get_missing_value_percentage(20)
greater_than_20


# ### Removing Null Values

# In[60]:


# function to drop cols which have more than 20% null values

def remove_cols_with_nulls (df, threshold):
    myCol = list(df.columns)
    for col in myCol: 
        percentage = (df[col].isnull().sum()/len(df[col]))*100
        if percentage>threshold:
            df.drop(col, axis=1, inplace=True)


# In[61]:


df.shape


# In[62]:


remove_cols_with_nulls(df, 20)


# In[63]:


df.shape


# In[64]:


len(df.columns[(df.isnull().any())])


# In[65]:


# Deleting rows with any null value
df.dropna(how='all',axis=0, inplace=True) 


# In[66]:


# Therefore, there is no row will all NULL values
df.shape


# In[67]:


# Deleting rows with any null value
df.dropna(how='any',axis=0, inplace=True) 


# In[68]:


df.shape


# In[69]:


msno.matrix(df)


# In[70]:


df.isnull().any().sum()


# In[71]:


df.columns


# In[72]:


df.select_dtypes('number')


# In[73]:


df['R&D Expenses'].value_counts()


# In[74]:


df.shape[0]


# In[75]:


df['R&D Expenses'].value_counts(normalize=True).sort_values(ascending=False)[0]


# In[76]:


# more than 50% value of R&D expense is 0. Remove numeric cols with dominant values


# In[77]:


df.operatingProfitMargin.value_counts().sort_values(ascending=False).iloc[0]


# In[78]:


#Code goes to except block when single value is met ---> which is SERIES and you cannot do simple indexing in SERIES


# In[79]:


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


# In[80]:


df.shape


# In[81]:


df.columns


# In[82]:


df.select_dtypes(include='number')


# In[83]:


df.select_dtypes(include='object')


# In[84]:


df.select_dtypes(include='number').shape


# In[85]:


df.select_dtypes(include='object').shape


# In[86]:


df.select_dtypes(include='category').shape


# In[87]:


df.shape


# In[88]:


# 155 cols ----> 154 number, 1 Object


# In[89]:


df.Sector.value_counts()


# In[90]:


sector_list = list(df.Sector.unique())


# In[91]:


sector_list


# In[92]:


pd.get_dummies(df.Sector, drop_first=True)


# In[93]:


Sector_status = pd.get_dummies(df.Sector, drop_first=True)

#Adding the result to the original housing dataframe

df = pd.concat([df, Sector_status], axis=1)


# In[94]:


# Droppig Sector Column as we are done with 
df.drop("Sector", axis=1, inplace=True)


# In[95]:


df.shape


# In[96]:


df.Energy.value_counts()


# In[97]:


# Sector Column will be dropped after Exploratory Data Analysis


# In[98]:


df.head()


# # Removing columns with single value

# In[99]:


def removeSingleValue (col):
    length = len(df[col].value_counts())
    if (length<2):
        print(col)
        df.drop(col, axis=1, inplace=True)


# In[100]:


for col in df.columns:
    removeSingleValue(col)


# In[101]:


num_col = list(df.dtypes[df.dtypes !='object'].index)


# In[102]:


len(num_col)


# In[103]:


df.shape


# ### Duplicate Row Checker

# In[104]:


df.duplicated().sum()


# In[105]:


# Moving "Class" Column to end
df['Result'] = df.Class
df.drop("Class", axis=1, inplace=True)
df = df.rename(columns={"Result":"Class"})


# In[ ]:





# In[106]:


df.head()


# # Outlier Treatment

# In[107]:


df.head()


# ## Method 1 Standard Deviation Method
# 
# Three standard deviations from the mean is a common cut-off in practice for identifying outliers in a Gaussian or Gaussian-like distribution. For smaller samples of data, perhaps a value of 2 standard deviations (95%) can be used, and for larger samples, perhaps a value of 4 standard deviations (99.9%) can be used.

# In[108]:


df.shape


# In[109]:


# Extracting numerical columns from the telecom_df data frame.
numerical_cols = df.select_dtypes(include = np.number).columns.to_list()


# In[110]:


# calculate summary statistics
data = df[numerical_cols]
data_mean, data_std = np.mean(data), np.std(data)
# identify outliers
cut_off = data_std * 2.5
lower, upper = data_mean - cut_off, data_mean + cut_off
# identify outliers
outliers = df[((df < lower) | (df > upper)).any(axis=1)]
print('Number of identified outliers: %d' % len(outliers))


# In[111]:


# remove outliers
outliers_removed = df[~((df < lower) | (df > upper)).any(axis=1)]
print('Non-outlier observations: %d' % len(outliers_removed))


# ## Method 2 IQR method
# 
# The IQR can be used to identify outliers by defining limits on the sample values that are a factor k of the IQR below the 25th percentile or above the 75th percentile. The common value for the factor k is the value 1.5.

# In[112]:


Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
print(IQR)


# In[113]:


outliers_removed_IQR = df[~((df < (Q1 - 1.5 * IQR)) |(df > (Q3 + 1.5 * IQR))).any(axis=1)]
print('Non-outlier observations: %d' % len(outliers_removed_IQR))


# ## Method 3:  99-1 percentile method

# In[114]:


Q1 = df.quantile(0.01)
Q3 = df.quantile(0.99)
IQR = Q3 - Q1
print(IQR)


# In[115]:


outliers_removed_IQR = df[~((df < (Q1 - 1.5 * IQR)) |(df > (Q3 + 1.5 * IQR))).any(axis=1)]
print('Non-outlier observations: %d' % len(outliers_removed_IQR))


# In[116]:


### We choose 99-1 percentile method for outlier treatment
df_99_1 = df[~((df < (Q1 - 1.5 * IQR)) |(df > (Q3 + 1.5 * IQR))).any(axis=1)]


# In[117]:


df_99_1.shape


# ## Method 4:  95-5 percentile method

# In[118]:


Q1 = df.quantile(0.05)
Q3 = df.quantile(0.95)
IQR = Q3 - Q1


# In[119]:


outliers_removed_IQR = df[~((df < (Q1 - 1.5 * IQR)) |(df > (Q3 + 1.5 * IQR))).any(axis=1)]
print('Non-outlier observations: %d' % len(outliers_removed_IQR))


# In[120]:


df_95_5 = df[~((df < (Q1 - 1.5 * IQR)) |(df > (Q3 + 1.5 * IQR))).any(axis=1)]


# In[121]:


df_95_5.shape


# # Observation
# 
# - Percentile method leads to poor performance later.
# 
# - Forced to go with 2.5 times the STD method after trial and error even though we are losing out on a lot of data

# In[122]:


df = outliers_removed


# In[123]:


df.reset_index(inplace=True)


# In[124]:


df.drop(['index'], axis=1, inplace=True)


# In[125]:


from sklearn.ensemble import IsolationForest


clf = IsolationForest()
preds = clf.fit_predict(df)


# In[126]:


len(preds)


# In[127]:


len(df)


# In[128]:


for i in range(0,len(preds)):
    if preds[i]==-1:
        df.drop(i, axis=0, inplace=True)


# In[129]:


len(df)


# ### Observation
# 
# - Isolation Forest Outlier Detection method further removed 4 data points

# In[130]:


df.drop('Class', axis = 1, inplace=True)


# In[ ]:





# In[132]:


df['Next_Year_Price_Var[%]']


# # Train-Test Split

# In[133]:


X = df.drop(['Next_Year_Price_Var[%]'], axis=1)
X.head()


# In[135]:


y = df['Next_Year_Price_Var[%]']
y.head()


# In[136]:


# split into train and test
df_train, df_test = train_test_split(df, train_size=0.7, test_size = 0.3, random_state=747)


# ## Dividing training dataset to X and Y for the model building
# ## Feature Scaling
# 

# In[137]:


# scaling the features

scaler = StandardScaler()
var = list(df_train.columns)
df_train[var] = scaler.fit_transform(df_train[var])

df_test[var] = scaler.transform(df_test[var])


# In[138]:


#pop will remove the column and return it to y_train
y_train = df_train.pop("Next_Year_Price_Var[%]")
X_train = df_train

y_test = df_test.pop("Next_Year_Price_Var[%]")
X_test = df_test


# In[139]:


X_train.head()


# # Data Modelling
# 
# ## Recursive Feature Elimination
# 
# - We will use RFE to remove 40% of columns
# - 125 * 0.60 = 108
# - We will keep 100 columns
# 
# 
# ## Note: 
# 
# Tried with 100 cols but many of their VIF value is INF and p-value is also high

# In[170]:


from sklearn.linear_model import LinearRegression


# In[171]:


# Creating the Linear Regression Model and running RFE to get top 20 columns

lm = LinearRegression()
lm.fit(X_train, y_train)

rfe = RFE(lm, 20)
rfe = rfe.fit(X_train, y_train)


# In[172]:


# Checking out the selected columns

top = X_train.columns[rfe.support_]
top


# In[173]:


# Assign the features to X_train_rfe

X_train_rfe = X_train[top]


# In[174]:


X_train_rfe


# In[175]:


# Associate the new  features to X_train and X_test for further analysis

X_train = X_train_rfe[X_train_rfe.columns]
X_test =  X_test[X_train.columns]


# # Building StatsModel

# In[176]:


# creating X_train dataframe with RFE selected top100 variables

X_train_rfe = X_train[top]


# In[177]:


import statsmodels.api as sm

#Adding constant
X_train_rfe = sm.add_constant(X_train_rfe)


# In[178]:


#Running the model
lm = sm.OLS(y_train, X_train_rfe).fit()


# In[179]:


lm.summary()


# In[180]:


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


# In[181]:


vif = vif_cal(X_train_rfe)
vif


# In[182]:


# Dropping Shareholders Equity per Share (p-value = 0.881)
dropCol = 'Shareholders Equity per Share'
X_train_rfe.drop(dropCol, axis=1, inplace=True)

lm = sm.OLS(y_train, X_train_rfe ).fit()

print(lm.summary())


# In[183]:


# Dropping EBIT (p-value = 0.861)
dropCol = 'EBIT'
X_train_rfe.drop(dropCol, axis=1, inplace=True)

lm = sm.OLS(y_train, X_train_rfe ).fit()

print(lm.summary())


# In[184]:


# Dropping Net Profit Margin (p-value = 0.861)
dropCol = 'Net Profit Margin'
X_train_rfe.drop(dropCol, axis=1, inplace=True)

lm = sm.OLS(y_train, X_train_rfe ).fit()

print(lm.summary())


# In[185]:


# Dropping EBITDA (p-value = 0.721)
dropCol = 'EBITDA'
X_train_rfe.drop(dropCol, axis=1, inplace=True)

lm = sm.OLS(y_train, X_train_rfe ).fit()

print(lm.summary())


# In[ ]:





# In[186]:


vif = vif_cal(X_train_rfe)
vif


# In[187]:



# Dropping EBITDA (VIF = INF)
dropCol = 'eBITperRevenue'
X_train_rfe.drop(dropCol, axis=1, inplace=True)

lm = sm.OLS(y_train, X_train_rfe ).fit()

print(lm.summary())


# In[ ]:





# In[188]:


X_train_rfe.drop("const", axis=1, inplace=True)


# In[189]:



# Associate the new features to X_train and X_test for further analysis

X_train = X_train_rfe[X_train_rfe.columns]
X_test =  X_test[X_train.columns]


# # Residual Analysis

# In[190]:


lm = LinearRegression()
lm.fit(X_train, y_train)


# In[191]:


y_train_pred = lm.predict(X_train)


# In[192]:


residual = y_train - y_train_pred


# In[193]:


plt.figure(figsize=(8,4))
ax = sns.distplot(residual)


plt.ylabel('Residual')
plt.xlabel('Errors')
plt.yticks([])
sns.despine(left=True)

plt.show()


# In[194]:


plt.figure(figsize=(8,4))
ax = sns.scatterplot(x=y_train_pred, y=residual)


plt.axhline(y=2.15, color='#FFBF00', linestyle='-')
plt.axhline(y=-1.65, color='#FFBF00', linestyle='-')

plt.ylabel('Residual')
plt.xlabel('Predictions')
plt.yticks([])
# plt.xticks([])
sns.despine(left=True)

plt.show()


# # Observation:
# - Residuals form a normal distribution with average around zero
# - Residuals are **NOT** independent of each other
# - Residuals have constant variance

# # Conclusion
# 
# Since not all assumptions made my LinearRegression stand, we cannot apply LinearRegression or trust its result.

# In[ ]:




