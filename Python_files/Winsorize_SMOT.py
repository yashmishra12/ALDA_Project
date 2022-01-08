#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import scipy
import seaborn as sns

import missingno as msno

# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)

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
# from xgboost import XGBClassifier

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
from imblearn.over_sampling import SMOTE, RandomOverSampler as smot
from imblearn.under_sampling import NearMiss, RandomUnderSampler
from imblearn.combine import SMOTEENN, SMOTETomek


# In[2]:


# !pip install lazypredict


# In[3]:


# !pip uninstall scikit-learn -y


# In[4]:


# !pip install scikit-learn==0.23.1


# In[5]:


# import lazypredict


# In[6]:


# from lazypredict.Supervised import LazyClassifier


# In[7]:


# from lazypredict.Supervised import LazyClassifier
# from sklearn.datasets import load_diabetes
# from sklearn.model_selection import train_test_split
# # from lazypredict.Supervised import LazyRegressor
# # from sklearn import datasets
# from sklearn.utils import shuffle
# # import numpy as np


# ### Reading CSV files

# In[8]:


df_2014 = pd.read_csv("2014_Financial_Data.csv")
df_2015 = pd.read_csv("2015_Financial_Data.csv")
df_2016 = pd.read_csv("2016_Financial_Data.csv")
df_2017 = pd.read_csv("2017_Financial_Data.csv")
df_2018 = pd.read_csv("2018_Financial_Data.csv")


# In[9]:


nasdaq = pd.read_csv("nasdaq.csv")


# In[10]:


nasdaq.shape


# In[11]:


nasdaq.columns


# In[12]:


nasdaq.head()


# In[13]:


nasdaq.drop('Name', axis=1, inplace=True)


# In[14]:


df_2014.shape


# In[15]:


df_2015.shape


# In[16]:


df_2016.shape


# In[17]:


df_2017.shape


# In[18]:


df_2018.shape


# In[19]:


df_2014['Year'] = 2014
df_2015['Year'] = 2015
df_2016['Year'] = 2016
df_2017['Year'] = 2017
df_2018['Year'] = 2018


# In[20]:


df_2014.columns


# In[21]:


df_2014.rename(columns={'2015 PRICE VAR [%]':'Next_Year_Price_Var[%]'}, inplace=True)
df_2015.rename(columns={'2016 PRICE VAR [%]':'Next_Year_Price_Var[%]'}, inplace=True)
df_2016.rename(columns={'2017 PRICE VAR [%]':'Next_Year_Price_Var[%]'}, inplace=True)
df_2017.rename(columns={'2018 PRICE VAR [%]':'Next_Year_Price_Var[%]'}, inplace=True)
df_2018.rename(columns={'2019 PRICE VAR [%]':'Next_Year_Price_Var[%]'}, inplace=True)


# In[22]:


df = pd.concat([df_2014, df_2015, df_2016, df_2017, df_2018], axis = 0)


# In[23]:


df.shape


# In[24]:


df.columns


# In[25]:


df.head()


# In[26]:


df.rename(columns={"Unnamed: 0": "Symbol"}, inplace=True)
# df_2014.rename(columns={'2015 PRICE VAR [%]':'Next_Year_Price_Var[%]'}, inplace=True)


# In[27]:


df = pd.merge(df, nasdaq, how="inner", on="Symbol")


# In[28]:


df.shape


# In[29]:


df.head()


# In[30]:


df.Country.value_counts()


# In[31]:


df.Country.isnull().sum()


# In[ ]:





# In[32]:


#Next_Year_Price_Var[%] +ve ---> class = 1, if -ve -----> class = 0
df.drop('Next_Year_Price_Var[%]', axis=1, inplace=True)


# In[33]:


df.rename(columns={"Symbol":"Name"}, inplace=True)


# In[34]:


df.Name.nunique()


# In[35]:


df.shape


# In[36]:


df.info


# In[37]:


df.describe()


# In[38]:


## Removing "Year" because our future companies will have different years and it should not affect our final call
df.drop("Year", axis=1, inplace=True)


# In[39]:


## Removing "Name" because our future companies will have different Name and it should not affect our final call
df.drop("Name", axis=1, inplace=True)


# In[40]:


df.head(2)


# In[41]:


df.loc[(df.Country=="United States")].shape


# In[42]:


df.loc[(df.Country=="United States")]


# In[43]:


# Selecting only United States


# In[44]:


df = df.loc[(df.Country=="United States")]


# In[45]:


df.drop('Country', axis=1, inplace=True)


# ### Understanding Null Value Distribution

# In[46]:


df.isnull().sum().sort_values(ascending=False)


# In[47]:


(df.isnull().sum() * 100 / len(df)).sort_values(ascending=False)


# In[48]:


(df.isnull().sum() * 100 / len(df)).sort_values(ascending=True).head(15)


# In[49]:


# all cols have some null values
len(df.isnull().any())


# In[50]:


# No columns with all null values
df.columns[df.isnull().all()]


# In[51]:


# Defining a funtion to add the count/frequency values as annotation to histogram.
def annotate_graph(ax):
    for bar in ax.patches:         
        ax.annotate(format((bar.get_height()), '.0f'),                    
                    (bar.get_x() + bar.get_width() / 2,  bar.get_height()),                    
                    ha='center', va='center',                    
                    size=10, xytext=(0, 8),                    
                    textcoords='offset points')
    return ax


# In[52]:


# Plotting histogram for the dataframe and columns having null values.
plt.figure(figsize=(28,10))

ax = sns.histplot(round((df.isnull().sum()/len(df.index) * 100).sort_values(ascending=False), 2))
ax = annotate_graph(ax)

ax.set(xticks=np.arange(0,101))
ax.set(xlabel='Null value percentage', ylabel='Count of columns with null values')
sns.despine()
plt.tight_layout()


# In[53]:


msno.matrix(df)


# In[54]:


# defining a function to get more than cutoff percent missing value

def get_missing_value_percentage(cutoff):
    y = pd.DataFrame( round((df.isnull().sum()/len(df.index) * 100).sort_values(ascending=False), 2))
    y.rename(columns={0:"Percentage"}, inplace=True)
    y2 = y[y.Percentage>cutoff]
    return y2


# In[55]:


# get columns with more than 70% missing values
greater_than_70 = get_missing_value_percentage(70)


# In[56]:


len(greater_than_70)


# In[57]:


greater_than_70


# In[58]:


# get columns with more than 50% missing values
greater_than_50 = get_missing_value_percentage(50)


# In[59]:


len(greater_than_50)


# In[60]:


greater_than_50


# In[61]:


# get columns with more than 20% missing values
greater_than_20 = get_missing_value_percentage(20)
greater_than_20


# ### Removing Null Values

# In[62]:


# function to drop cols which have more than 20% null values

def remove_cols_with_nulls (df, threshold):
    myCol = list(df.columns)
    for col in myCol: 
        percentage = (df[col].isnull().sum()/len(df[col]))*100
        if percentage>threshold:
            df.drop(col, axis=1, inplace=True)


# In[63]:


df.shape


# In[64]:


remove_cols_with_nulls(df, 20)


# In[65]:


df.shape


# In[66]:


len(df.columns[(df.isnull().any())])


# In[67]:


# Deleting rows with any null value
df.dropna(how='all',axis=0, inplace=True) 


# In[68]:


# Therefore, there is no row will all NULL values
df.shape


# In[69]:


# Deleting rows with any null value
df.dropna(how='any',axis=0, inplace=True) 


# In[70]:


df.shape


# In[71]:


msno.matrix(df)


# In[72]:


df.isnull().any().sum()


# In[73]:


df.columns


# In[74]:


df.select_dtypes('number')


# In[75]:


df['R&D Expenses'].value_counts()


# In[76]:


df.shape[0]


# In[77]:


df['R&D Expenses'].value_counts(normalize=True).sort_values(ascending=False)[0]


# In[78]:


# more than 50% value of R&D expense is 0. Remove numeric cols with dominant values


# In[79]:


df.operatingProfitMargin.value_counts().sort_values(ascending=False).iloc[0]


# In[80]:


#Code goes to except block when single value is met ---> which is SERIES and you cannot do simple indexing in SERIES


# In[81]:


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


# In[82]:


df.shape


# In[83]:


df.columns


# In[84]:


df.select_dtypes(include='number')


# In[85]:


df.select_dtypes(include='object')


# In[86]:


df.select_dtypes(include='number').shape


# In[87]:


df.select_dtypes(include='object').shape


# In[88]:


df.select_dtypes(include='category').shape


# In[89]:


df.shape


# In[90]:


# 155 cols ----> 154 number, 1 Object


# In[91]:


df.Sector.value_counts()


# In[92]:


sector_list = list(df.Sector.unique())


# In[93]:


sector_list


# In[94]:


pd.get_dummies(df.Sector, drop_first=True)


# In[95]:


Sector_status = pd.get_dummies(df.Sector, drop_first=True)

#Adding the result to the original housing dataframe

df = pd.concat([df, Sector_status], axis=1)


# In[96]:


# Droppig Sector Column as we are done with 
df.drop("Sector", axis=1, inplace=True)


# In[97]:


df.shape


# In[98]:


df.Energy.value_counts()


# In[99]:


# Sector Column will be dropped after Exploratory Data Analysis


# In[100]:


df.head()


# # Removing columns with single value

# In[101]:


def removeSingleValue (col):
    length = len(df[col].value_counts())
    if (length<2):
        print(col)
        df.drop(col, axis=1, inplace=True)


# In[102]:


for col in df.columns:
    removeSingleValue(col)


# In[103]:


num_col = list(df.dtypes[df.dtypes !='object'].index)


# In[104]:


len(num_col)


# In[105]:


df.shape


# ### Duplicate Row Checker

# In[106]:


df.duplicated().sum()


# In[107]:


# Moving "Class" Column to end
df['Result'] = df.Class
df.drop("Class", axis=1, inplace=True)
df = df.rename(columns={"Result":"Class"})


# In[ ]:





# In[108]:


df.head()


# # Outlier Treatment

# In[109]:


df.head()


# ## Method 1 Standard Deviation Method
# 
# Three standard deviations from the mean is a common cut-off in practice for identifying outliers in a Gaussian or Gaussian-like distribution. For smaller samples of data, perhaps a value of 2 standard deviations (95%) can be used, and for larger samples, perhaps a value of 4 standard deviations (99.9%) can be used.

# In[110]:


df.shape


# In[111]:


# Extracting numerical columns from the telecom_df data frame.
numerical_cols = df.select_dtypes(include = np.number).columns.to_list()


# In[112]:


# calculate summary statistics
data = df[numerical_cols]
data_mean, data_std = np.mean(data), np.std(data)
# identify outliers
cut_off = data_std * 3
lower, upper = data_mean - cut_off, data_mean + cut_off
# identify outliers
outliers = df[((df < lower) | (df > upper)).any(axis=1)]
print('Number of identified outliers: %d' % len(outliers))


# In[113]:


# remove outliers
outliers_removed = df[~((df < lower) | (df > upper)).any(axis=1)]
print('Non-outlier observations: %d' % len(outliers_removed))


# ## Method 2 IQR method
# 
# The IQR can be used to identify outliers by defining limits on the sample values that are a factor k of the IQR below the 25th percentile or above the 75th percentile. The common value for the factor k is the value 1.5.

# In[114]:


Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
print(IQR)


# In[115]:


outliers_removed_IQR = df[~((df < (Q1 - 1.5 * IQR)) |(df > (Q3 + 1.5 * IQR))).any(axis=1)]
print('Non-outlier observations: %d' % len(outliers_removed_IQR))


# ## Method 3:  99-1 percentile method

# In[116]:


Q1 = df.quantile(0.01)
Q3 = df.quantile(0.99)
IQR = Q3 - Q1
print(IQR)


# In[117]:


outliers_removed_IQR = df[~((df < (Q1 - 1.5 * IQR)) |(df > (Q3 + 1.5 * IQR))).any(axis=1)]
print('Non-outlier observations: %d' % len(outliers_removed_IQR))


# In[118]:


### We choose 99-1 percentile method for outlier treatment
df_99_1 = df[~((df < (Q1 - 1.5 * IQR)) |(df > (Q3 + 1.5 * IQR))).any(axis=1)]


# In[119]:


df_99_1.shape


# ## Method 4:  95-5 percentile method

# In[120]:


Q1 = df.quantile(0.05)
Q3 = df.quantile(0.95)
IQR = Q3 - Q1


# In[121]:


outliers_removed_IQR = df[~((df < (Q1 - 1.5 * IQR)) |(df > (Q3 + 1.5 * IQR))).any(axis=1)]
print('Non-outlier observations: %d' % len(outliers_removed_IQR))


# In[122]:


### We choose 95-5 percentile method for outlier treatment
df_95_5 = df[~((df < (Q1 - 1.5 * IQR)) |(df > (Q3 + 1.5 * IQR))).any(axis=1)]


# In[123]:


df_95_5.shape


# In[124]:


# Selecting 99-1 percentile


# In[125]:


df = df_99_1


# # Observation
# 
# We are losing a lot of data even if we perform 99-1 percentile outlier removal. Therefore, we need to explore other techniques

# In[126]:


df.head()


# In[127]:


(df.isnull().sum() * 100 / len(df)).sort_values(ascending=False)


# In[128]:


df.shape


# # Winsorize

# In[129]:


sorted(sector_list)


# In[130]:


df_winow = pd.DataFrame()
for col in df:
    if (col not in sector_list):
        df_winow[col] = winsorize(df[col], (0.1, 0.1))
    else:
        df_winow[col] = df[col].values


# In[131]:


df_winow.shape


# In[132]:


df['Communication Services'].value_counts()


# In[133]:


df_winow.shape


# In[134]:


df_winow


# In[135]:


df_winow.Utilities.value_counts()


# In[136]:


df_winow.Energy.value_counts()


# In[137]:


hig_neg_corr = list(df_winow.corr()['Class'].sort_values(ascending=True).index[0:5])

fig = plt.figure(figsize=(20,12))

ax0=fig.add_subplot(2,2,1)
ax1=fig.add_subplot(2,2,2)
ax2=fig.add_subplot(2,2,3)
ax3=fig.add_subplot(2,2,4)

sns.boxplot(df_winow[hig_neg_corr[0]], hue="Class", data=df_winow, ax=ax0)
sns.boxplot(df_winow[hig_neg_corr[1]], hue="Class", data=df_winow, ax=ax1)
sns.boxplot(df_winow[hig_neg_corr[2]], hue="Class", data=df_winow, ax=ax2)
sns.boxplot(df_winow[hig_neg_corr[3]], hue="Class", data=df_winow, ax=ax3)

sns.despine()
plt.tight_layout()
plt.plot()


# In[138]:


hig_pos_corr = list(df_winow.corr()['Class'].sort_values(ascending=False).index[1:5])

fig = plt.figure(figsize=(20,12))

ax0=fig.add_subplot(2,2,1)
ax1=fig.add_subplot(2,2,2)
ax2=fig.add_subplot(2,2,3)
ax3=fig.add_subplot(2,2,4)

sns.boxplot(df_winow[hig_pos_corr[0]], hue="Class", data=df_winow, ax=ax0)
sns.boxplot(df_winow[hig_pos_corr[1]], hue="Class", data=df_winow, ax=ax1)
sns.boxplot(df_winow[hig_pos_corr[2]], hue="Class", data=df_winow, ax=ax2)
sns.boxplot(df_winow[hig_pos_corr[3]], hue="Class", data=df_winow, ax=ax3)

sns.despine()
plt.tight_layout()
plt.plot()


# In[ ]:





# In[ ]:





# ## 15 percentile - 85 percentile

# In[139]:


df_winow2 = pd.DataFrame()
for col in df:
    if(col not in sector_list):
        df_winow2[col] = winsorize(df[col], (0.15, 0.15))
    if(col in sector_list):
        df_winow2[col] = df[col].values


# In[140]:


df_winow2.shape


# In[141]:


hig_neg_corr = list(df_winow2.corr()['Class'].sort_values(ascending=True).index[0:5])

fig = plt.figure(figsize=(20,12))

ax0=fig.add_subplot(2,2,1)
ax1=fig.add_subplot(2,2,2)
ax2=fig.add_subplot(2,2,3)
ax3=fig.add_subplot(2,2,4)

sns.boxplot(df_winow2[hig_neg_corr[0]], hue="Class", data=df_winow2, ax=ax0)
sns.boxplot(df_winow2[hig_neg_corr[1]], hue="Class", data=df_winow2, ax=ax1)
sns.boxplot(df_winow2[hig_neg_corr[2]], hue="Class", data=df_winow2, ax=ax2)
sns.boxplot(df_winow2[hig_neg_corr[3]], hue="Class", data=df_winow2, ax=ax3)

sns.despine()
plt.tight_layout()
plt.plot()


# In[142]:


hig_pos_corr = list(df_winow2.corr()['Class'].sort_values(ascending=False).index[1:5])

fig = plt.figure(figsize=(20,12))

ax0=fig.add_subplot(2,2,1)
ax1=fig.add_subplot(2,2,2)
ax2=fig.add_subplot(2,2,3)
ax3=fig.add_subplot(2,2,4)

sns.boxplot(df_winow2[hig_pos_corr[0]], hue="Class", data=df_winow2, ax=ax0)
sns.boxplot(df_winow2[hig_pos_corr[1]], hue="Class", data=df_winow2, ax=ax1)
sns.boxplot(df_winow2[hig_pos_corr[2]], hue="Class", data=df_winow2, ax=ax2)
sns.boxplot(df_winow2[hig_pos_corr[3]], hue="Class", data=df_winow2, ax=ax3)

sns.despine()
plt.tight_layout()
plt.plot()


# In[ ]:





# In[ ]:





# In[143]:


from sklearn.feature_selection import VarianceThreshold

var_thr = VarianceThreshold(threshold = 0.15) #Removing both constant and quasi-constant
var_thr.fit(df_winow2)

var_thr.get_support()


# In[144]:


concol = [column for column in df_winow2.columns 
          if column not in df_winow2.columns[var_thr.get_support()]]

for features in concol:
    print(features)


# In[145]:


len(concol)


# In[146]:


df_winow2.shape


# In[147]:


# df_winow2.drop(concol, axis=1, inplace=True)


# # Observation
# 
# We get a balanced data with Winsorize 15-85

# In[148]:


df = df_winow2


# ## Train Test Split

# In[149]:


my_cv = 5


# In[150]:


X = df.drop('Class', axis = 1)
y = df[['Class']]

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.8, test_size = 0.2, random_state = 123)

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


# In[151]:


y_train_reshape = pd.DataFrame(y_train.values.reshape(-1,1))
print("Counts of label '1': {}".format((y_train_reshape==1).sum()[0]))
print("Counts of label '0': {} \n".format((y_train_reshape==0).sum()[0]))

y_train_1 = (y_train_reshape==1).sum()[0]
print("Percentage of Profitable Company : {}% \n".format(round(y_train_1/len(y_train_reshape)*100,2)))


# # Scalers

# In[152]:


#Importing the PCA module

pca = PCA(random_state=42)
pca_again = PCA(0.95)


# ## MinMax Scaler

# In[153]:


X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.8, test_size = 0.2, random_state = 123)


# In[154]:


scaler = MinMaxScaler()


X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[155]:


X_train_pca_mm = pca_again.fit_transform(X_train)


# In[156]:


X_train_pca_mm.shape


# In[157]:


# Tranforming X_Test
X_test_pca_mm = pca_again.transform(X_test)
X_test_pca_mm.shape


# In[158]:


#Doing the PCA on the train data
pca.fit(X_train)


# In[159]:


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





# In[160]:


pca = PCA(n_components=5)


# In[161]:


scaler = MinMaxScaler()


# In[162]:


scaled_df=df_winow2.copy()
scaled_df=pd.DataFrame(scaler.fit_transform(scaled_df), columns=scaled_df.columns)
pca_fit = pca.fit(scaled_df)


# In[ ]:





# In[163]:


fig = plt.figure(figsize=(14,5))
PC_values = np.arange(pca_fit.n_components_) + 1

plt.plot(PC_values, pca_fit.explained_variance_ratio_, 'o-', linewidth=2, color='blue')
plt.title('Scree Plot')
plt.xlabel('Principal Component')
plt.ylabel('Variance Explained')

plt.show()


# In[164]:


print(pca_fit.explained_variance_ratio_)


# ## StandardScaler

# In[ ]:





# In[165]:


#Importing the PCA module

pca = PCA(random_state=42)
pca_again = PCA(0.95)


# In[166]:


X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.8, test_size = 0.2, random_state = 123)


# In[167]:


scaler = StandardScaler()


X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[168]:


X_train_pca_std = pca_again.fit_transform(X_train)


# In[169]:


X_train_pca_std.shape


# In[170]:


# Tranforming X_Test
X_test_pca_std = pca_again.transform(X_test)
X_test_pca_std.shape


# In[171]:


#Doing the PCA on the train data
pca.fit(X_train)


# In[172]:


#Making the screeplot - plotting the cumulative variance against the number of components
get_ipython().run_line_magic('matplotlib', 'inline')
fig = plt.figure(figsize = (12,8))
plt.vlines(x = X_train_pca_std.shape[1],ymax = 1,ymin = 0,colors = 'r',linestyles = '--')
plt.hlines(y = 0.95, xmax = 70,xmin = 0,colors = 'g',linestyles = '--')
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')

sns.despine()
plt.show()


# In[ ]:





# In[173]:


pca = PCA(n_components=5)


# In[174]:


scaler = StandardScaler()


# In[175]:


scaled_df=df_winow2.copy()
scaled_df=pd.DataFrame(scaler.fit_transform(scaled_df), columns=scaled_df.columns)
pca_fit = pca.fit(scaled_df)


# In[ ]:





# In[176]:


fig = plt.figure(figsize=(14,5))
PC_values = np.arange(pca_fit.n_components_) + 1

plt.plot(PC_values, pca_fit.explained_variance_ratio_, 'o-', linewidth=2, color='blue')
plt.title('Scree Plot')
plt.xlabel('Principal Component')
plt.ylabel('Variance Explained')

plt.show()


# In[177]:


print(pca_fit.explained_variance_ratio_)


# In[ ]:





# # Data Preparation for Modelling

# In[178]:


X = df_winow2.drop('Class', axis = 1)
y = df_winow2[['Class']]

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.8, test_size = 0.2, random_state = 100, stratify=y)

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


# In[179]:


# Normalize the data 

scaler = MinMaxScaler()

scaled_data = scaler.fit_transform(X_train)

X_train = pd.DataFrame(data = scaled_data, index = X_train.index, columns = X_train.columns)
X_test = pd.DataFrame(data = scaler.transform(X_test), index = X_test.index, columns = X_test.columns)


# In[ ]:





# In[180]:


print("Before OverSampling, counts of label '1': {}".format((y_train==1).sum()[0]))
print("Before OverSampling, counts of label '0': {} \n".format((y_train==0).sum()[0]))

y_train_1 = (y_train==1).sum()[0]
print("Before OverSampling, churn event rate : {}% \n".format(round(y_train_1/len(y_train)*100,2)))


# In[181]:


sm_smot = smot(random_state=27, sampling_strategy=1)
X_train_res, y_train_res = sm_smot.fit_resample(X_train, y_train)
X_train_res = X_train
y_train_res = y_train


# # Defining functions for Modelling

# In[182]:


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


# In[183]:


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


# In[184]:


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
    


# In[185]:


# Defining function to write the evaluation metrics of the models into data frame.
def WriteModelMetrics(series_metrics,metricsdataframe):
    metricsdataframe = metricsdataframe.append(series_metrics,ignore_index=True)
    return metricsdataframe


# In[186]:


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


# In[187]:


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


# In[188]:


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


# In[189]:


# This method will perform cross-validation and the display the model report.
def modelfit(alg, X_train, y_train, performCV=True, cv_folds=my_cv):
    #Fit the algorithm on the data
    alg.fit(X_train, y_train)
        
    #Predict training set:
    dtrain_predictions = alg.predict(X_train)
    dtrain_predprob = alg.predict_proba(X_train)[:,1]
    
    #Perform cross-validation:
    if performCV:
        cv_score = cross_val_score(alg, X_train, y_train, cv=cv_folds, scoring='accuracy')
    
    #Print model report:
    print ("\nModel Report")
    print ("Accuracy : %.3g" % metrics.roc_auc_score(y_train, dtrain_predictions))
    print ("Recall/Sensitivity : %.3g" % metrics.recall_score(y_train, dtrain_predictions))
    print ("AUC Score (Train): %f" % metrics.roc_auc_score(y_train, dtrain_predprob))
    
    if performCV:
        print ("CV Score : Mean - %.3g | Std - %.3g | Min - %.3g | Max - %.3g" % (np.mean(cv_score),np.std(cv_score),np.min(cv_score),np.max(cv_score)))
        


# In[190]:


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


# In[191]:


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

# In[192]:


X_train = X_train_pca_mm
X_test = X_test_pca_mm


# In[193]:


X_train_pca = X_train
y_train_res = y_train
X_test_pca = X_test
# y_test = y_test

y_train_res=y_train_res.values.reshape(-1,1)
# y_test = y_test.values.reshape(-1,1)


# In[194]:


X_train_pca.shape


# In[195]:


y_train_res.shape


# In[196]:


X_test_pca.shape


# In[197]:


y_test.shape


# In[198]:


y_train_res = pd.DataFrame(y_train_res)


# In[199]:


y_train_res.rename(columns={0:"Class"}, inplace=True)


# In[200]:


y_train_res


# In[201]:


y_test 


# # 1. Logistic Regression

# In[202]:


# Creating Train-Test variables for Logistic Regression
X_train_lr = pd.DataFrame(X_train_pca)
y_train_lr = pd.DataFrame(y_train_res)
X_test_lr = pd.DataFrame(X_test_pca)
y_test_lr = y_test


# In[203]:


logml = sm.GLM(y_train_lr, (sm.add_constant(X_train_lr)), family = sm.families.Binomial())
logml.fit().summary()


# In[204]:


# Checking for the VIF of the train data.
vif = vif_cal(X_train_lr) 
vif


# ### Observation
# - There are no Columns which are highly correlated ---> VIF = 1
# - There are not many columns whose coefficients are not statistically significant ----> p>0.05

# In[205]:


lg = LogisticRegression()


# In[206]:


modelfit(lg, X_train_lr, y_train_lr)


# In[207]:


# predictions on Test data
pred_probs_test = lg.predict(X_test_lr)
getModelMetrics(y_test_lr,pred_probs_test)


# In[208]:


print("Accuracy : {}".format(metrics.accuracy_score(y_test_lr,pred_probs_test)))
print("Recall : {}".format(metrics.recall_score(y_test_lr,pred_probs_test)))
print("Precision : {}".format(metrics.precision_score(y_test_lr,pred_probs_test)))


# In[209]:


print(metrics.confusion_matrix(y_test_lr,pred_probs_test))


# In[210]:


#Making prediction on the test data
pred_probs_train = lg.predict_proba(X_train_lr)[:,1]

print("roc_auc_score(Train) {:2.2}".format(metrics.roc_auc_score(y_train_lr, pred_probs_train)))


# In[211]:


y_train_lr.rename(columns={0:"Class"}, inplace=True)

cut_off_prob=0.5
y_train_df,series_metrics = predictClassWithProb(lg,X_train_lr,y_train_lr,cut_off_prob)


# In[212]:


draw_roc(y_train_df.Class, y_train_df.final_predicted)
print("roc_auc_score : {:2.2f}".format(metrics.roc_auc_score(y_train_df.Class, y_train_df.final_predicted)))


# In[213]:


#draw_roc(y_pred_final.Churn, y_pred_final.predicted)
print("roc_auc_score : {:2.2f}".format(metrics.roc_auc_score(y_train_df.Class, y_train_df.final_predicted)))


# In[214]:


# finding cut-off with the right balance of the metrices
# sensitivity vs specificity trade-off
findOptimalCutoff(y_train_df)


# In[215]:


# predicting with the choosen cut-off on TRAIN
cut_off_prob = 0.5
y_train_df,series_metrics = predictClassWithProb(lg,X_train_lr,y_train_lr,cut_off_prob,model_name='Logistic Regression',train_or_test='TRAIN')
metricsdataframe=WriteModelMetrics(series_metrics,metricsdataframe)


# In[216]:


### predicting with the choosen cut-off on TEST
cut_off_prob=0.5
y_train_df,series_metrics = predictClassWithProb(lg,X_test_lr,y_test_lr,cut_off_prob,model_name='Logistic Regression',train_or_test='TEST')
metricsdataframe=WriteModelMetrics(series_metrics,metricsdataframe)


# #### 

# ## 2. Decision Tree

# In[217]:


# Creating Train-Test variables for Decision Tree
X_train_dt = pd.DataFrame(X_train_pca)
y_train_dt = pd.DataFrame(y_train_res)
X_test_dt = pd.DataFrame(X_test_pca)
y_test_dt = y_test


# In[218]:


X_train_dt.shape


# In[219]:


y_train_dt.shape


# In[220]:


X_test_dt.shape, y_test_dt.shape


# In[221]:


##### Applying Decision Tree Classifier on our principal components with Hyperparameter tuning
dt = DecisionTreeClassifier(class_weight='balanced',
                             max_features='auto',
                             min_samples_split=100,
                             min_samples_leaf=100,
                             max_depth=10,
                             random_state=123)

modelfit(dt, X_train_dt, y_train_dt)


# In[222]:


# make predictions
pred_probs_test = dt.predict(X_test_dt)

#Let's check the model metrices.

getModelMetrics(y_test_dt,pred_probs_test)


# In[223]:


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
grid_search = GridSearchCV(estimator = dt, param_grid = param_grid, cv = my_cv, n_jobs = -1,verbose = 1000,scoring="f1_weighted")


# In[224]:


# Fit the grid search to the data
grid_search.fit(X_train_dt, y_train_dt)


# In[225]:


# printing the optimal accuracy score and hyperparameters
print('We can get score of',grid_search.best_score_,'using',grid_search.best_params_)


# In[226]:


cv_df = pd.DataFrame(grid_search.cv_results_)
cv_df.head(3)


# In[227]:


cv_df.nlargest(3,"mean_test_score")


# In[228]:


grid_search.best_score_


# In[229]:


grid_search.best_estimator_


# In[230]:


param_max_depth = cv_df.nlargest(3,"mean_test_score").param_max_depth.iloc[0]
param_max_features = cv_df.nlargest(3,"mean_test_score").param_max_features.iloc[0]
param_min_samples_leaf = cv_df.nlargest(3,"mean_test_score").param_min_samples_leaf.iloc[0]
param_min_samples_split = cv_df.nlargest(3,"mean_test_score").param_min_samples_split.iloc[0]


# In[231]:


# model with the best hyperparameters
dt_final = DecisionTreeClassifier(class_weight='balanced',
                             max_depth=param_max_depth,
                             max_features=param_max_features,
                             min_samples_leaf=param_min_samples_leaf, 
                             min_samples_split=param_min_samples_split,
                             random_state=123)


# In[232]:


modelfit(dt_final,X_train_dt,y_train_dt)


# In[233]:


draw_roc(y_train_df.Class, y_train_df.final_predicted)
print("roc_auc_score : {:2.2f}".format(metrics.roc_auc_score(y_train_df.Class, y_train_df.final_predicted)))


# In[234]:


# make predictions
pred_probs_test = dt_final.predict(X_test_dt)
#Let's check the model metrices.
getModelMetrics(actual_Class=y_test_dt,pred_Class=pred_probs_test)


# In[235]:


# predicting churn with default cut-off 0.5
cut_off_prob = 0.5
y_train_df,series_metrics = predictClassWithProb(dt_final,X_train_dt,y_train_dt,cut_off_prob)


# In[236]:


# finding cut-off with the right balance of the metrices
findOptimalCutoff(y_train_df)


# In[237]:


# predicting churn with cut-off 0.4
cut_off_prob=0.36
y_train_df,series_metrics = predictClassWithProb(dt_final,X_train_dt,y_train_dt,cut_off_prob,model_name='Decision Tree',train_or_test='TRAIN')
metricsdataframe=WriteModelMetrics(series_metrics,metricsdataframe)


# In[238]:


#Lets see how it performs on test data.
y_test_df,series_metrics= predictClassWithProb(dt_final,X_test_dt,y_test_dt,cut_off_prob,model_name='Decision Tree',train_or_test='TEST')
metricsdataframe=WriteModelMetrics(series_metrics,metricsdataframe)


# # 3. Random Forest

# In[239]:


# Creating Train-Test variables for Random Forest
X_train_rf = pd.DataFrame(X_train_pca)
y_train_rf = pd.DataFrame(y_train_res)
X_test_rf = pd.DataFrame(X_test_pca)
y_test_rf = y_test


# In[240]:


rf = RandomForestClassifier(random_state=42, max_depth=5, n_estimators=20, oob_score=True)


# In[241]:


rf.fit(X_train_rf, y_train_rf)


# In[242]:


rf.oob_score_


# In[243]:


# make predictions
pred_probs_test = rf.predict(X_test_rf)

#Let's check the model metrices.
getModelMetrics(actual_Class=y_test_rf,pred_Class=pred_probs_test)


# In[244]:


parameters = {'max_depth': range(5, 40, 5)}
rf = RandomForestClassifier()
grid_search = GridSearchCV(rf, parameters, cv=my_cv, scoring="f1_weighted", verbose=1000, return_train_score=True)

grid_search.fit(X_train_rf, y_train_rf)


# In[245]:


pd.DataFrame(grid_search.cv_results_).sort_values(by="rank_test_score")


# In[246]:


# grid_search.cv_results_

plot_traintestAcc(grid_search.cv_results_,'max_depth')


# In[247]:


my_max_depth = pd.DataFrame(grid_search.cv_results_).sort_values(by="rank_test_score")['param_max_depth'].iloc[0]


# ### Tuning n_estimators

# In[248]:


parameters = {'n_estimators': range(5, 70, 5)}

rf = RandomForestClassifier(max_depth=my_max_depth,random_state=10)
grid_search = GridSearchCV(rf, parameters, cv=my_cv, scoring="f1_weighted", verbose=100, return_train_score=True)


grid_search.fit(X_train_rf, y_train_rf)


# In[249]:


pd.DataFrame(grid_search.cv_results_).sort_values(by="rank_test_score").head()


# In[250]:


random_forst_feature_graph(grid_search, "n_estimators")


# In[251]:


my_n_estimator = pd.DataFrame(grid_search.cv_results_).sort_values(by="rank_test_score")['param_n_estimators'].iloc[0]


# ### Tuning max_features

# In[252]:


parameters = {'max_features': [5, 10, 15, 20, 25, 30,50,70]}

rf = RandomForestClassifier(max_depth=my_max_depth,n_estimators = my_n_estimator,random_state=10)
grid_search = GridSearchCV(rf, parameters, cv=my_cv, scoring="f1_weighted", verbose=1000, return_train_score=True)


grid_search.fit(X_train_rf, y_train_rf)


# In[253]:


pd.DataFrame(grid_search.cv_results_).sort_values(by="rank_test_score")


# In[254]:


random_forst_feature_graph(grid_search, "max_features")


# In[255]:


my_max_features=pd.DataFrame(grid_search.cv_results_).sort_values(by="rank_test_score")['param_max_features'].iloc[0]


# ### Tuning min_samples_leaf

# In[256]:


parameters = {'min_samples_leaf': range(1, 500, 50)}

rf = RandomForestClassifier(max_depth=my_max_depth,n_estimators = my_n_estimator, max_features = my_max_features, random_state=10)
grid_search = GridSearchCV(rf, parameters, cv=my_cv, scoring="f1_weighted", verbose=1000, return_train_score=True)

grid_search.fit(X_train_rf, y_train_rf)


# In[257]:


pd.DataFrame(grid_search.cv_results_).sort_values(by="rank_test_score")


# In[258]:


random_forst_feature_graph(grid_search, "min_samples_leaf")


# In[259]:


my_min_sample_leaf = pd.DataFrame(grid_search.cv_results_).sort_values(by="rank_test_score")['param_min_samples_leaf'].iloc[0]


# ### Tuning min_samples_split

# In[260]:


parameters = {'min_samples_split': range(50, 550, 50)}

rf = RandomForestClassifier(max_depth=my_max_depth,n_estimators = my_n_estimator, max_features = my_max_features, min_samples_leaf = my_min_sample_leaf,random_state=10)
grid_search = GridSearchCV(rf, parameters, cv=my_cv, scoring="f1_weighted", verbose=1000, return_train_score=True)


grid_search.fit(X_train_rf, y_train_rf)


# In[261]:


pd.DataFrame(grid_search.cv_results_).sort_values(by="rank_test_score")


# In[262]:


random_forst_feature_graph(grid_search, "min_samples_split")


# In[263]:


my_min_samples_split=pd.DataFrame(grid_search.cv_results_).sort_values(by="rank_test_score")['param_min_samples_split'].iloc[0]


# ### Final Model

# In[264]:


rf_final = RandomForestClassifier(max_depth=my_max_depth,
                                  n_estimators = my_n_estimator, 
                                  max_features = my_max_features, 
                                  min_samples_leaf = my_min_sample_leaf,
                                  min_samples_split=my_min_samples_split,
                                  random_state=123)


# In[265]:


print("Model performance on Train data:")
modelfit(rf_final,X_train_rf,y_train_rf)


# In[266]:


# predict on test data
predictions = rf_final.predict(X_test_rf)


# In[267]:


print("Model performance on Test data:")
getModelMetrics(y_test_rf,predictions)


# In[268]:


# predicting churn with default cut-off 0.5
cut_off_prob=0.5
y_train_df,series_metrics = predictClassWithProb(rf_final,X_train_rf,y_train_rf,cut_off_prob)


# In[269]:


# finding cut-off with the right balance of the metrices
findOptimalCutoff(y_train_df)


# In[270]:


## Training Data 
cut_off_prob=0.25

y_train_df,series_metrics=predictClassWithProb(rf_final,X_train_rf,y_train_rf,cut_off_prob,model_name='Random Forest',train_or_test='TRAIN')
metricsdataframe=WriteModelMetrics(series_metrics,metricsdataframe)
y_train_df.head()


# In[271]:


# Testing Data
y_test_df,series_metrics= predictClassWithProb(rf_final,X_test_rf,y_test_rf,cut_off_prob,model_name='Random Forest',train_or_test='TEST')
metricsdataframe=WriteModelMetrics(series_metrics,metricsdataframe)


# # Conclusion
# 
# - We are getting Low Accuracy (in late 50s) and low precision (in late 50s) which is too poor to be acceptable

# In[ ]:




