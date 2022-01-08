#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import scale 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df=pd.read_csv("naya_df.csv")


# In[3]:


df.head()


# In[4]:


df.drop(['Unnamed: 0', 'Country'], axis=1, inplace=True)


# In[5]:



# split into train and test
df_train, df_test = train_test_split(df, train_size=0.7, test_size = 0.3, random_state=747)


# In[6]:


# scaling the features
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
var = list(df_train.columns)
df_train[var] = scaler.fit_transform(df_train[var])

df_test[var] = scaler.transform(df_test[var])


# In[7]:


#pop will remove the column and return it to y_train
y_train = df_train.pop("Class")
X_train = df_train

y_test = df_test.pop("Class")
X_test = df_test


# In[24]:


lm = LinearRegression()
lm.fit(X_train, y_train)

rfe = RFE(lm, 70)
rfe = rfe.fit(X_train, y_train)


# In[25]:


lasso = Lasso()

# list of alphas

# params = {'alpha': [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 
#                     9.0, 10.0, 20, 50, 100, 500, 1000 ]}


params ={'alpha': [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.001, 0.002, 0.003, 0.004, 0.005, 0.01]} 
#Commented the first params because it gives same optimum alpha of 0.0001 but this range gives better resulting graph



# cross validation

folds = 5
lasso_model_cv = GridSearchCV(estimator = lasso,                         
                        param_grid = params, 
                        scoring= 'neg_mean_absolute_error', 
                        cv = folds, 
                        return_train_score=True,
                        verbose = 1)            

lasso_model_cv.fit(X_train, y_train)


# In[26]:


# display the mean scores

lasso_cv_results = pd.DataFrame(lasso_model_cv.cv_results_)
lasso_cv_results[['param_alpha', 'mean_train_score', 'mean_test_score', 'rank_test_score']].sort_values(by = ['rank_test_score'])


# In[27]:


# plotting mean test and train scoes with alpha 

lasso_cv_results['param_alpha'] = lasso_cv_results['param_alpha'].astype('float64')

# plotting
plt.figure(figsize=(16,8))
plt.plot(lasso_cv_results['param_alpha'], lasso_cv_results['mean_train_score'])
plt.plot(lasso_cv_results['param_alpha'], lasso_cv_results['mean_test_score'])
plt.xlabel('alpha')
plt.ylabel('neg_mean_absolute_error')

sns.despine()
plt.legend(['train score', 'test score'], loc='upper right')
plt.show()


# In[28]:



# get the best estimator for lambda

lasso_model_cv.best_estimator_


# In[29]:


# check the coefficient values with lambda = 0.0001

alpha = 0.0001

lasso = Lasso(alpha=alpha)
        
lasso.fit(X_train, y_train) 
lasso.coef_


# In[30]:


# Check the mean squared error

mean_squared_error(y_test, lasso.predict(X_test))


# In[31]:



# Put the shortlisted Features and coefficienst in a dataframe

lasso_df = pd.DataFrame({'Features':X_train.columns, 'Coefficient':lasso.coef_.round(2)})

print("Shape before zero coefficients are removed: ",lasso_df.shape)

lasso_df = lasso_df[lasso_df['Coefficient'] != 0.00]

print("Shape after zero coefficients are removed:  ",lasso_df.shape)

lasso_df.reset_index(drop=True, inplace=True)
lasso_df


# In[32]:


# Put the Features and Coefficients in dictionary

lasso_coeff_dict = dict(pd.Series(lasso.coef_, index = X_train.columns))
lasso_coeff_dict


# In[33]:


# Method to get the coefficient values

def find(x):
    return lasso_coeff_dict[x]


X_train_lasso = X_train[lasso_df.Features]


# In[34]:



lasso_temp_df = pd.DataFrame(list(zip( X_train_lasso.columns, rfe.support_, rfe.ranking_)), columns=['Features', 'rfe_support', 'rfe_ranking'])
lasso_temp_df = lasso_temp_df.loc[lasso_temp_df['rfe_support'] == True]
lasso_temp_df.reset_index(drop=True, inplace=True)

lasso_temp_df['Coefficient'] = lasso_temp_df['Features'].apply(find)
lasso_temp_df = lasso_temp_df.sort_values(by=['Coefficient'], ascending=False)
lasso_temp_df


# In[56]:


# bar plot to determine the variables that would affect pricing most using ridge regression

plt.figure(figsize=(10,5))
lasso_temp_df_pos = lasso_temp_df.copy()
lasso_temp_df_pos['Coefficient'] = abs(lasso_temp_df_pos['Coefficient'] )

sns.barplot(y = 'Features', x='Coefficient',  data = lasso_temp_df_pos.head(5)).set(xlabel='Lasso: Absolute Coefficient')
sns.despine()

plt.show()


# # Ridge

# In[38]:


# list of alphas

params = {'alpha': [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 
                    9.0, 10.0, 20, 50, 100, 500, 1000 ]}

ridge = Ridge()

# cross validation

folds = 5
ridge_model_cv = GridSearchCV(estimator = ridge, 
                        param_grid = params, 
                        scoring= 'neg_mean_absolute_error',  #r2 can also be used (both give similar lambda in our case)
                        cv = folds, 
                        return_train_score=True,
                        verbose = 1)            
ridge_model_cv.fit(X_train, y_train)


# In[39]:


# display the mean scores

ridge_cv_results = pd.DataFrame(ridge_model_cv.cv_results_)
ridge_cv_results = ridge_cv_results[ridge_cv_results['param_alpha']<=1000]
ridge_cv_results[['param_alpha', 'mean_train_score', 'mean_test_score', 'rank_test_score']].sort_values(by = ['rank_test_score'])


# In[40]:


# plotting mean test and train scoes with alpha 

ridge_cv_results['param_alpha'] = ridge_cv_results['param_alpha'].astype('int32')

# plotting
plt.figure(figsize=(16,8))
plt.plot(ridge_cv_results['param_alpha'], ridge_cv_results['mean_train_score'])
plt.plot(ridge_cv_results['param_alpha'], ridge_cv_results['mean_test_score'])
plt.xlabel('alpha')
plt.ylabel('neg_mean_absolute_error')

sns.despine()
plt.legend(['train score', 'test score'], loc='upper right')
plt.show()


# In[41]:


# get the best estimator for lambda

ridge_model_cv.best_params_


# In[42]:


# check the coefficient values with lambda = 1

ridge = Ridge(alpha=0.1)

ridge.fit(X_train, y_train)
ridge.coef_


# In[43]:



# Check the mean squared error

mean_squared_error(y_test, ridge.predict(X_test))


# In[44]:


# Put the Features and coefficienst in a dataframe

ridge_df = pd.DataFrame({'Features':X_train.columns, 'Coefficient':ridge.coef_.round(4)})
ridge_df.reset_index(drop=True, inplace=True)
print("Ridge DataFrame Shape:", ridge_df.shape)
ridge_df.head(10)


# In[46]:


#Assign the Features and their coefficient values to a dictionary which would be used while plotting the bar plot

ridge_coeff_dict = dict(pd.Series(ridge.coef_.round(4), index = X_train.columns))
ridge_coeff_dict


# In[47]:


X_train_ridge = X_train[ridge_df.Features]


# In[48]:


# Method to get the coefficient values

def find(x):
    return ridge_coeff_dict[x]

# Assign top features to a temp dataframe for further display in the bar plot

ridge_temp_df = pd.DataFrame(list(zip( X_train.columns, rfe.support_, rfe.ranking_)), columns=['Features', 'rfe_support', 'rfe_ranking'])
ridge_temp_df = ridge_temp_df.loc[ridge_temp_df['rfe_support'] == True]
ridge_temp_df.reset_index(drop=True, inplace=True)

ridge_temp_df['Coefficient'] = ridge_temp_df['Features'].apply(find)
ridge_temp_df = ridge_temp_df.sort_values(by=['Coefficient'], ascending=False)
ridge_temp_df


# In[55]:


# bar plot to determine the variables that would affect pricing most using ridge regression

ridge_temp_df_pos = ridge_temp_df.copy()
ridge_temp_df_pos['Coefficient'] = abs(ridge_temp_df_pos['Coefficient'] )

plt.figure(figsize=(10,5))
sns.barplot(y = 'Features', x='Coefficient', data = ridge_temp_df_pos.head(5)).set(xlabel='Ridge: Absolute Coefficient')
sns.despine()
plt.show()


# In[ ]:




