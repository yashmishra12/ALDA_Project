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


get_ipython().system('pip uninstall scikit-learn -y')
# # 


# In[ ]:


get_ipython().system('pip install scikit-learn==0.23.1')


# 
# 
# ```
# 
# ```
# 
# # Sentimental Analysis

# In[ ]:


import pandas as pd
import datetime
dfAllTweets = pd.read_csv('drive/MyDrive/ALDA data/Tweet.csv')
dfAllCompanies= pd.read_csv('drive/MyDrive/ALDA data/Company_Tweet.csv')


# In[ ]:


df_merged= pd.merge(dfAllTweets, dfAllCompanies, on="tweet_id")


# In[ ]:


df_merged.head


# In[ ]:


df_merged['post_date'] = pd.to_datetime(df_merged['post_date'], unit='s')
df_merged['year'] = pd.DatetimeIndex(df_merged['post_date']).year


# In[ ]:


pip install textblob


# APPLE 2015

# In[ ]:


df_2015_apple = df_merged[df_merged['ticker_symbol']=='AAPL']
df_2015_apple = df_2015_apple.loc[df_merged['year']==2015]
df_2015_apple= df_2015_apple[:5000]

from textblob import TextBlob

analysis_list= []
count_pos= 0
count_neg= 0
count_nt= 0
for i in range(len(df_2015_apple)):
  # print(i)
  analysis = TextBlob(df_2015_apple.iloc[i]['body'])
  likes= df_2015_apple.iloc[i]['like_num']+1
  retweets=df_2015_apple.iloc[i]['retweet_num']+1
  res=''
  if analysis.sentiment.polarity > 0:
      count_pos+=likes+retweets
      res= 'positive'
  elif analysis.sentiment.polarity == 0:
      count_nt+=likes+retweets
      res= 'neutral'
  else:
      count_neg+=likes+retweets
      res= 'negative'
  analysis_list.append(res)

print('positive tweets ', count_pos)
print('negative tweets ', count_neg)
print('neutral tweets ', count_nt)
if count_pos> count_neg:
   print("overall sentiment: positive")
else:
  print("overall sentiment: negative")

df_2015_apple['sentiment']= analysis_list
df_2015_apple.to_csv('drive/MyDrive/ALDA data/2015_apple.csv', encoding='utf-8', index=False)


# APPLE 2016

# In[ ]:


df_2016_apple = df_merged[df_merged['ticker_symbol']=='AAPL']
df_2016_apple = df_2016_apple.loc[df_merged['year']==2016]
df_2016_apple= df_2016_apple[:5000]

from textblob import TextBlob

analysis_list= []
count_pos= 0
count_neg= 0
count_nt= 0
for i in range(len(df_2016_apple)):
  # print(i)
  analysis = TextBlob(df_2016_apple.iloc[i]['body'])
  likes= df_2016_apple.iloc[i]['like_num']+1
  retweets=df_2016_apple.iloc[i]['retweet_num']+1
  res=''
  if analysis.sentiment.polarity > 0:
      count_pos+=likes+retweets
      res= 'positive'
  elif analysis.sentiment.polarity == 0:
      count_nt+=likes+retweets
      res= 'neutral'
  else:
      count_neg+=likes+retweets
      res= 'negative'
  analysis_list.append(res)

print('positive tweets ', count_pos)
print('negative tweets ', count_neg)
print('neutral tweets ', count_nt)
if count_pos> count_neg:
   print("overall sentiment: positive")
else:
  print("overall sentiment: negative")

df_2016_apple['sentiment']= analysis_list
df_2016_apple.to_csv('drive/MyDrive/ALDA data/2016_apple.csv', encoding='utf-8', index=False)


# APPLE 2017

# In[ ]:


df_2017_apple = df_merged[df_merged['ticker_symbol']=='AAPL']
df_2017_apple = df_2017_apple.loc[df_merged['year']==2017]
df_2017_apple= df_2017_apple[:5000]

from textblob import TextBlob

analysis_list= []
count_pos= 0
count_neg= 0
count_nt= 0
for i in range(len(df_2017_apple)):
  # print(i)
  analysis = TextBlob(df_2017_apple.iloc[i]['body'])
  likes= df_2017_apple.iloc[i]['like_num']+1
  retweets=df_2017_apple.iloc[i]['retweet_num']+1
  res=''
  if analysis.sentiment.polarity > 0:
      count_pos+=likes+retweets
      res= 'positive'
  elif analysis.sentiment.polarity == 0:
      count_nt+=likes+retweets
      res= 'neutral'
  else:
      count_neg+=likes+retweets
      res= 'negative'
  analysis_list.append(res)

print('positive tweets ', count_pos)
print('negative tweets ', count_neg)
print('neutral tweets ', count_nt)
if count_pos> count_neg:
   print("overall sentiment: positive")
else:
  print("overall sentiment: negative")

df_2017_apple['sentiment']= analysis_list
df_2017_apple.to_csv('drive/MyDrive/ALDA data/2017_apple.csv', encoding='utf-8', index=False)


# APPLE 2018

# In[ ]:


df_2018_apple = df_merged[df_merged['ticker_symbol']=='AAPL']
df_2018_apple = df_2018_apple.loc[df_merged['year']==2018]
df_2018_apple= df_2018_apple[:5000]

from textblob import TextBlob

analysis_list= []
count_pos= 0
count_neg= 0
count_nt= 0
for i in range(len(df_2018_apple)):
  # print(i)
  analysis = TextBlob(df_2018_apple.iloc[i]['body'])
  likes= df_2018_apple.iloc[i]['like_num']+1
  retweets=df_2018_apple.iloc[i]['retweet_num']+1
  res=''
  if analysis.sentiment.polarity > 0:
      count_pos+=likes+retweets
      res= 'positive'
  elif analysis.sentiment.polarity == 0:
      count_nt+=likes+retweets
      res= 'neutral'
  else:
      count_neg+=likes+retweets
      res= 'negative'
  analysis_list.append(res)

print('positive tweets ', count_pos)
print('negative tweets ', count_neg)
print('neutral tweets ', count_nt)
if count_pos> count_neg:
   print("overall sentiment: positive")
else:
  print("overall sentiment: negative")

df_2018_apple['sentiment']= analysis_list
df_2018_apple.to_csv('drive/MyDrive/ALDA data/2018_apple.csv', encoding='utf-8', index=False)


# AMAZON 2015

# In[ ]:


df_2015_amazon = df_merged[df_merged['ticker_symbol']=='AMZN']
df_2015_amazon = df_2015_amazon.loc[df_merged['year']==2015]
df_2015_amazon= df_2015_amazon[:5000]

from textblob import TextBlob

analysis_list= []
count_pos= 0
count_neg= 0
count_nt= 0
for i in range(len(df_2015_amazon)):
  # print(i)
  analysis = TextBlob(df_2015_amazon.iloc[i]['body'])
  likes= df_2015_amazon.iloc[i]['like_num']+1
  retweets=df_2015_amazon.iloc[i]['retweet_num']+1
  res=''
  if analysis.sentiment.polarity > 0:
      count_pos+=likes+retweets
      res= 'positive'
  elif analysis.sentiment.polarity == 0:
      count_nt+=likes+retweets
      res= 'neutral'
  else:
      count_neg+=likes+retweets
      res= 'negative'
  analysis_list.append(res)

print('positive tweets ', count_pos)
print('negative tweets ', count_neg)
print('neutral tweets ', count_nt)
if count_pos> count_neg:
   print("overall sentiment: positive")
else:
  print("overall sentiment: negative")

df_2015_amazon['sentiment']= analysis_list
df_2015_amazon.to_csv('drive/MyDrive/ALDA data/2015_amazon.csv', encoding='utf-8', index=False)


# AMAZON 2016

# In[ ]:


df_2016_amazon = df_merged[df_merged['ticker_symbol']=='AMZN']
df_2016_amazon = df_2016_amazon.loc[df_merged['year']==2016]
df_2016_amazon= df_2016_amazon[:5000]

from textblob import TextBlob

analysis_list= []
count_pos= 0
count_neg= 0
count_nt= 0
for i in range(len(df_2016_amazon)):
  # print(i)
  analysis = TextBlob(df_2016_amazon.iloc[i]['body'])
  likes= df_2016_amazon.iloc[i]['like_num']+1
  retweets=df_2016_amazon.iloc[i]['retweet_num']+1
  res=''
  if analysis.sentiment.polarity > 0:
      count_pos+=likes+retweets
      res= 'positive'
  elif analysis.sentiment.polarity == 0:
      count_nt+=likes+retweets
      res= 'neutral'
  else:
      count_neg+=likes+retweets
      res= 'negative'
  analysis_list.append(res)

print('positive tweets ', count_pos)
print('negative tweets ', count_neg)
print('neutral tweets ', count_nt)
if count_pos> count_neg:
   print("overall sentiment: positive")
else:
  print("overall sentiment: negative")

df_2016_amazon['sentiment']= analysis_list
df_2016_amazon.to_csv('drive/MyDrive/ALDA data/2016_amazon.csv', encoding='utf-8', index=False)


# AMAZON 2017

# In[ ]:


df_2017_amazon = df_merged[df_merged['ticker_symbol']=='AMZN']
df_2017_amazon = df_2017_amazon.loc[df_merged['year']==2017]
df_2017_amazon= df_2017_amazon[:5000]

from textblob import TextBlob

analysis_list= []
count_pos= 0
count_neg= 0
count_nt= 0
for i in range(len(df_2017_amazon)):
  # print(i)
  analysis = TextBlob(df_2017_amazon.iloc[i]['body'])
  likes= df_2017_amazon.iloc[i]['like_num']+1
  retweets=df_2017_amazon.iloc[i]['retweet_num']+1
  res=''
  if analysis.sentiment.polarity > 0:
      count_pos+=likes+retweets
      res= 'positive'
  elif analysis.sentiment.polarity == 0:
      count_nt+=likes+retweets
      res= 'neutral'
  else:
      count_neg+=likes+retweets
      res= 'negative'
  analysis_list.append(res)

print('positive tweets ', count_pos)
print('negative tweets ', count_neg)
print('neutral tweets ', count_nt)
if count_pos> count_neg:
   print("overall sentiment: positive")
else:
  print("overall sentiment: negative")

df_2017_amazon['sentiment']= analysis_list
df_2017_amazon.to_csv('drive/MyDrive/ALDA data/2017_amazon.csv', encoding='utf-8', index=False)


# AMAZON 2018

# In[ ]:


df_2018_amazon = df_merged[df_merged['ticker_symbol']=='AMZN']
df_2018_amazon = df_2018_amazon.loc[df_merged['year']==2018]
df_2018_amazon= df_2018_amazon[:5000]

from textblob import TextBlob

analysis_list= []
count_pos= 0
count_neg= 0
count_nt= 0
for i in range(len(df_2018_amazon)):
  # print(i)
  analysis = TextBlob(df_2018_amazon.iloc[i]['body'])
  likes= df_2018_amazon.iloc[i]['like_num']+1
  retweets=df_2018_amazon.iloc[i]['retweet_num']+1
  res=''
  if analysis.sentiment.polarity > 0:
      count_pos+=likes+retweets
      res= 'positive'
  elif analysis.sentiment.polarity == 0:
      count_nt+=likes+retweets
      res= 'neutral'
  else:
      count_neg+=likes+retweets
      res= 'negative'
  analysis_list.append(res)

print('positive tweets ', count_pos)
print('negative tweets ', count_neg)
print('neutral tweets ', count_nt)
if count_pos> count_neg:
   print("overall sentiment: positive")
else:
  print("overall sentiment: negative")

df_2018_amazon['sentiment']= analysis_list
df_2018_amazon.to_csv('drive/MyDrive/ALDA data/2018_amazon.csv', encoding='utf-8', index=False)


#  MICROSOFT 2015

# In[ ]:


df_2015_microsoft = df_merged[df_merged['ticker_symbol']=='MSFT']
df_2015_microsoft = df_2015_microsoft.loc[df_merged['year']==2015]
df_2015_microsoft= df_2015_microsoft[:5000]

from textblob import TextBlob

analysis_list= []
count_pos= 0
count_neg= 0
count_nt= 0
for i in range(len(df_2015_microsoft)):
  # print(i)
  analysis = TextBlob(df_2015_microsoft.iloc[i]['body'])
  likes= df_2015_microsoft.iloc[i]['like_num']+1
  retweets=df_2015_microsoft.iloc[i]['retweet_num']+1
  res=''
  if analysis.sentiment.polarity > 0:
      count_pos+=likes+retweets
      res= 'positive'
  elif analysis.sentiment.polarity == 0:
      count_nt+=likes+retweets
      res= 'neutral'
  else:
      count_neg+=likes+retweets
      res= 'negative'
  analysis_list.append(res)

print('positive tweets ', count_pos)
print('negative tweets ', count_neg)
print('neutral tweets ', count_nt)
if count_pos> count_neg:
   print("overall sentiment: positive")
else:
  print("overall sentiment: negative")

df_2015_microsoft['sentiment']= analysis_list
df_2015_microsoft.to_csv('drive/MyDrive/ALDA data/2015_microsoft.csv', encoding='utf-8', index=False)


#  MICROSOFT 2016

# In[ ]:


df_2016_microsoft = df_merged[df_merged['ticker_symbol']=='MSFT']
df_2016_microsoft = df_2016_microsoft.loc[df_merged['year']==2016]
df_2016_microsoft= df_2016_microsoft[:5000]

from textblob import TextBlob

analysis_list= []
count_pos= 0
count_neg= 0
count_nt= 0
for i in range(len(df_2016_microsoft)):
  # print(i)
  analysis = TextBlob(df_2016_microsoft.iloc[i]['body'])
  likes= df_2016_microsoft.iloc[i]['like_num']+1
  retweets=df_2016_microsoft.iloc[i]['retweet_num']+1
  res=''
  if analysis.sentiment.polarity > 0:
      count_pos+=likes+retweets
      res= 'positive'
  elif analysis.sentiment.polarity == 0:
      count_nt+=likes+retweets
      res= 'neutral'
  else:
      count_neg+=likes+retweets
      res= 'negative'
  analysis_list.append(res)

print('positive tweets ', count_pos)
print('negative tweets ', count_neg)
print('neutral tweets ', count_nt)
if count_pos> count_neg:
   print("overall sentiment: positive")
else:
  print("overall sentiment: negative")

df_2016_microsoft['sentiment']= analysis_list
df_2016_microsoft.to_csv('drive/MyDrive/ALDA data/2016_microsoft.csv', encoding='utf-8', index=False)


#  MICROSOFT 2017

# In[ ]:


df_2017_microsoft = df_merged[df_merged['ticker_symbol']=='MSFT']
df_2017_microsoft = df_2017_microsoft.loc[df_merged['year']==2017]
df_2017_microsoft= df_2017_microsoft[:5000]

from textblob import TextBlob

analysis_list= []
count_pos= 0
count_neg= 0
count_nt= 0
for i in range(len(df_2017_microsoft)):
  # print(i)
  analysis = TextBlob(df_2017_microsoft.iloc[i]['body'])
  likes= df_2017_microsoft.iloc[i]['like_num']+1
  retweets=df_2017_microsoft.iloc[i]['retweet_num']+1
  res=''
  if analysis.sentiment.polarity > 0:
      count_pos+=likes+retweets
      res= 'positive'
  elif analysis.sentiment.polarity == 0:
      count_nt+=likes+retweets
      res= 'neutral'
  else:
      count_neg+=likes+retweets
      res= 'negative'
  analysis_list.append(res)

print('positive tweets ', count_pos)
print('negative tweets ', count_neg)
print('neutral tweets ', count_nt)
if count_pos> count_neg:
   print("overall sentiment: positive")
else:
  print("overall sentiment: negative")

df_2017_microsoft['sentiment']= analysis_list
df_2017_microsoft.to_csv('drive/MyDrive/ALDA data/2017_microsoft.csv', encoding='utf-8', index=False)


#  MICROSOFT 2018

# In[ ]:


df_2018_microsoft = df_merged[df_merged['ticker_symbol']=='MSFT']
df_2018_microsoft = df_2018_microsoft.loc[df_merged['year']==2018]
df_2018_microsoft= df_2018_microsoft[:5000]

from textblob import TextBlob

analysis_list= []
count_pos= 0
count_neg= 0
count_nt= 0
for i in range(len(df_2018_microsoft)):
  # print(i)
  analysis = TextBlob(df_2018_microsoft.iloc[i]['body'])
  likes= df_2018_microsoft.iloc[i]['like_num']+1
  retweets=df_2018_microsoft.iloc[i]['retweet_num']+1
  res=''
  if analysis.sentiment.polarity > 0:
      count_pos+=likes+retweets
      res= 'positive'
  elif analysis.sentiment.polarity == 0:
      count_nt+=likes+retweets
      res= 'neutral'
  else:
      count_neg+=likes+retweets
      res= 'negative'
  analysis_list.append(res)

print('positive tweets ', count_pos)
print('negative tweets ', count_neg)
print('neutral tweets ', count_nt)
if count_pos> count_neg:
   print("overall sentiment: positive")
else:
  print("overall sentiment: negative")

df_2018_microsoft['sentiment']= analysis_list
df_2018_microsoft.to_csv('drive/MyDrive/ALDA data/2018_microsoft.csv', encoding='utf-8', index=False)


#  TESLA 2015

# In[ ]:


df_2015_tesla = df_merged[df_merged['ticker_symbol']=='TSLA']
df_2015_tesla = df_2015_tesla.loc[df_merged['year']==2015]
df_2015_tesla= df_2015_tesla[:5000]

from textblob import TextBlob

analysis_list= []
count_pos= 0
count_neg= 0
count_nt= 0
for i in range(len(df_2015_tesla)):
  # print(i)
  analysis = TextBlob(df_2015_tesla.iloc[i]['body'])
  likes= df_2015_tesla.iloc[i]['like_num']+1
  retweets=df_2015_tesla.iloc[i]['retweet_num']+1
  res=''
  if analysis.sentiment.polarity > 0:
      count_pos+=likes+retweets
      res= 'positive'
  elif analysis.sentiment.polarity == 0:
      count_nt+=likes+retweets
      res= 'neutral'
  else:
      count_neg+=likes+retweets
      res= 'negative'
  analysis_list.append(res)

print('positive tweets ', count_pos)
print('negative tweets ', count_neg)
print('neutral tweets ', count_nt)
if count_pos> count_neg:
   print("overall sentiment: positive")
else:
  print("overall sentiment: negative")

df_2015_tesla['sentiment']= analysis_list
df_2015_tesla.to_csv('drive/MyDrive/ALDA data/2015_tesla.csv', encoding='utf-8', index=False)


#  TESLA 2016

# In[ ]:


df_2016_tesla = df_merged[df_merged['ticker_symbol']=='TSLA']
df_2016_tesla = df_2016_tesla.loc[df_merged['year']==2016]
df_2016_tesla= df_2016_tesla[:5000]

from textblob import TextBlob

analysis_list= []
count_pos= 0
count_neg= 0
count_nt= 0
for i in range(len(df_2016_tesla)):
  # print(i)
  analysis = TextBlob(df_2016_tesla.iloc[i]['body'])
  likes= df_2016_tesla.iloc[i]['like_num']+1
  retweets=df_2016_tesla.iloc[i]['retweet_num']+1
  res=''
  if analysis.sentiment.polarity > 0:
      count_pos+=likes+retweets
      res= 'positive'
  elif analysis.sentiment.polarity == 0:
      count_nt+=likes+retweets
      res= 'neutral'
  else:
      count_neg+=likes+retweets
      res= 'negative'
  analysis_list.append(res)

print('positive tweets ', count_pos)
print('negative tweets ', count_neg)
print('neutral tweets ', count_nt)
if count_pos> count_neg:
   print("overall sentiment: positive")
else:
  print("overall sentiment: negative")

df_2016_tesla['sentiment']= analysis_list
df_2016_tesla.to_csv('drive/MyDrive/ALDA data/2016_tesla.csv', encoding='utf-8', index=False)


#  TESLA 2017

# In[ ]:


df_2017_tesla = df_merged[df_merged['ticker_symbol']=='TSLA']
df_2017_tesla = df_2017_tesla.loc[df_merged['year']==2017]
df_2017_tesla= df_2017_tesla[:5000]

from textblob import TextBlob

analysis_list= []
count_pos= 0
count_neg= 0
count_nt= 0
for i in range(len(df_2017_tesla)):
  # print(i)
  analysis = TextBlob(df_2017_tesla.iloc[i]['body'])
  likes= df_2017_tesla.iloc[i]['like_num']+1
  retweets=df_2017_tesla.iloc[i]['retweet_num']+1
  res=''
  if analysis.sentiment.polarity > 0:
      count_pos+=likes+retweets
      res= 'positive'
  elif analysis.sentiment.polarity == 0:
      count_nt+=likes+retweets
      res= 'neutral'
  else:
      count_neg+=likes+retweets
      res= 'negative'
  analysis_list.append(res)

print('positive tweets ', count_pos)
print('negative tweets ', count_neg)
print('neutral tweets ', count_nt)
if count_pos> count_neg:
   print("overall sentiment: positive")
else:
  print("overall sentiment: negative")

df_2017_tesla['sentiment']= analysis_list
df_2017_tesla.to_csv('drive/MyDrive/ALDA data/2017_tesla.csv', encoding='utf-8', index=False)


#  TESLA 2018

# In[ ]:


df_2018_tesla = df_merged[df_merged['ticker_symbol']=='TSLA']
df_2018_tesla = df_2018_tesla.loc[df_merged['year']==2018]
df_2018_tesla= df_2018_tesla[:5000]

from textblob import TextBlob

analysis_list= []
count_pos= 0
count_neg= 0
count_nt= 0
for i in range(len(df_2018_tesla)):
  # print(i)
  analysis = TextBlob(df_2018_tesla.iloc[i]['body'])
  likes= df_2018_tesla.iloc[i]['like_num']+1
  retweets=df_2018_tesla.iloc[i]['retweet_num']+1
  res=''
  if analysis.sentiment.polarity > 0:
      count_pos+=likes+retweets
      res= 'positive'
  elif analysis.sentiment.polarity == 0:
      count_nt+=likes+retweets
      res= 'neutral'
  else:
      count_neg+=likes+retweets
      res= 'negative'
  analysis_list.append(res)

print('positive tweets ', count_pos)
print('negative tweets ', count_neg)
print('neutral tweets ', count_nt)
if count_pos> count_neg:
   print("overall sentiment: positive")
else:
  print("overall sentiment: negative")

df_2018_tesla['sentiment']= analysis_list
df_2018_tesla.to_csv('drive/MyDrive/ALDA data/2018_tesla.csv', encoding='utf-8', index=False)


#  ALPHABET-A 2015

# In[ ]:


df_2015_google1 = df_merged[df_merged['ticker_symbol']=='GOOG']
df_2015_google1 = df_2015_google1.loc[df_merged['year']==2015]
df_2015_google1= df_2015_google1[:5000]

from textblob import TextBlob

analysis_list= []
count_pos= 0
count_neg= 0
count_nt= 0
for i in range(len(df_2015_google1)):
  # print(i)
  analysis = TextBlob(df_2015_google1.iloc[i]['body'])
  likes= df_2015_google1.iloc[i]['like_num']+1
  retweets=df_2015_google1.iloc[i]['retweet_num']+1
  res=''
  if analysis.sentiment.polarity > 0:
      count_pos+=likes+retweets
      res= 'positive'
  elif analysis.sentiment.polarity == 0:
      count_nt+=likes+retweets
      res= 'neutral'
  else:
      count_neg+=likes+retweets
      res= 'negative'
  analysis_list.append(res)

print('positive tweets ', count_pos)
print('negative tweets ', count_neg)
print('neutral tweets ', count_nt)
if count_pos> count_neg:
   print("overall sentiment: positive")
else:
  print("overall sentiment: negative")

df_2015_google1['sentiment']= analysis_list
df_2015_google1.to_csv('drive/MyDrive/ALDA data/2015_google1.csv', encoding='utf-8', index=False)


#  ALPHABET-A 2016

# In[ ]:


df_2016_google1 = df_merged[df_merged['ticker_symbol']=='GOOG']
df_2016_google1 = df_2016_google1.loc[df_merged['year']==2016]
df_2016_google1= df_2016_google1[:5000]

from textblob import TextBlob

analysis_list= []
count_pos= 0
count_neg= 0
count_nt= 0
for i in range(len(df_2016_google1)):
  # print(i)
  analysis = TextBlob(df_2016_google1.iloc[i]['body'])
  likes= df_2016_google1.iloc[i]['like_num']+1
  retweets=df_2016_google1.iloc[i]['retweet_num']+1
  res=''
  if analysis.sentiment.polarity > 0:
      count_pos+=likes+retweets
      res= 'positive'
  elif analysis.sentiment.polarity == 0:
      count_nt+=likes+retweets
      res= 'neutral'
  else:
      count_neg+=likes+retweets
      res= 'negative'
  analysis_list.append(res)

print('positive tweets ', count_pos)
print('negative tweets ', count_neg)
print('neutral tweets ', count_nt)
if count_pos> count_neg:
   print("overall sentiment: positive")
else:
  print("overall sentiment: negative")

df_2016_google1['sentiment']= analysis_list
df_2016_google1.to_csv('drive/MyDrive/ALDA data/2016_google1.csv', encoding='utf-8', index=False)


#  ALPHABET-A 2017

# In[ ]:


df_2017_google1 = df_merged[df_merged['ticker_symbol']=='GOOG']
df_2017_google1 = df_2017_google1.loc[df_merged['year']==2017]
df_2017_google1= df_2017_google1[:5000]

from textblob import TextBlob

analysis_list= []
count_pos= 0
count_neg= 0
count_nt= 0
for i in range(len(df_2017_google1)):
  # print(i)
  analysis = TextBlob(df_2017_google1.iloc[i]['body'])
  likes= df_2017_google1.iloc[i]['like_num']+1
  retweets=df_2017_google1.iloc[i]['retweet_num']+1
  res=''
  if analysis.sentiment.polarity > 0:
      count_pos+=likes+retweets
      res= 'positive'
  elif analysis.sentiment.polarity == 0:
      count_nt+=likes+retweets
      res= 'neutral'
  else:
      count_neg+=likes+retweets
      res= 'negative'
  analysis_list.append(res)

print('positive tweets ', count_pos)
print('negative tweets ', count_neg)
print('neutral tweets ', count_nt)
if count_pos> count_neg:
   print("overall sentiment: positive")
else:
  print("overall sentiment: negative")

df_2017_google1['sentiment']= analysis_list
df_2017_google1.to_csv('drive/MyDrive/ALDA data/2017_google1.csv', encoding='utf-8', index=False)


#  ALPHABET-A 2018

# In[ ]:


df_2018_google1 = df_merged[df_merged['ticker_symbol']=='GOOG']
df_2018_google1 = df_2018_google1.loc[df_merged['year']==2018]
df_2018_google1= df_2018_google1[:5000]

from textblob import TextBlob

analysis_list= []
count_pos= 0
count_neg= 0
count_nt= 0
for i in range(len(df_2018_google1)):
  # print(i)
  analysis = TextBlob(df_2018_google1.iloc[i]['body'])
  likes= df_2018_google1.iloc[i]['like_num']+1
  retweets=df_2018_google1.iloc[i]['retweet_num']+1
  res=''
  if analysis.sentiment.polarity > 0:
      count_pos+=likes+retweets
      res= 'positive'
  elif analysis.sentiment.polarity == 0:
      count_nt+=likes+retweets
      res= 'neutral'
  else:
      count_neg+=likes+retweets
      res= 'negative'
  analysis_list.append(res)

print('positive tweets ', count_pos)
print('negative tweets ', count_neg)
print('neutral tweets ', count_nt)
if count_pos> count_neg:
   print("overall sentiment: positive")
else:
  print("overall sentiment: negative")

df_2018_google1['sentiment']= analysis_list
df_2018_google1.to_csv('drive/MyDrive/ALDA data/2018_google1.csv', encoding='utf-8', index=False)


#  ALPHABET-C 2015

# In[ ]:


df_2015_google2 = df_merged[df_merged['ticker_symbol']=='GOOGL']
df_2015_google2 = df_2015_google2.loc[df_merged['year']==2015]
df_2015_google2= df_2015_google2[:5000]

from textblob import TextBlob

analysis_list= []
count_pos= 0
count_neg= 0
count_nt= 0
for i in range(len(df_2015_google2)):
  # print(i)
  analysis = TextBlob(df_2015_google2.iloc[i]['body'])
  likes= df_2015_google2.iloc[i]['like_num']+1
  retweets=df_2015_google2.iloc[i]['retweet_num']+1
  res=''
  if analysis.sentiment.polarity > 0:
      count_pos+=likes+retweets
      res= 'positive'
  elif analysis.sentiment.polarity == 0:
      count_nt+=likes+retweets
      res= 'neutral'
  else:
      count_neg+=likes+retweets
      res= 'negative'
  analysis_list.append(res)

print('positive tweets ', count_pos)
print('negative tweets ', count_neg)
print('neutral tweets ', count_nt)
if count_pos> count_neg:
   print("overall sentiment: positive")
else:
  print("overall sentiment: negative")

df_2015_google2['sentiment']= analysis_list
df_2015_google2.to_csv('drive/MyDrive/ALDA data/2015_google2.csv', encoding='utf-8', index=False)


#  ALPHABET-C 2016

# In[ ]:


df_2016_google2 = df_merged[df_merged['ticker_symbol']=='GOOGL']
df_2016_google2 = df_2016_google2.loc[df_merged['year']==2016]
df_2016_google2= df_2016_google2[:5000]

from textblob import TextBlob

analysis_list= []
count_pos= 0
count_neg= 0
count_nt= 0
for i in range(len(df_2016_google2)):
  # print(i)
  analysis = TextBlob(df_2016_google2.iloc[i]['body'])
  likes= df_2016_google2.iloc[i]['like_num']+1
  retweets=df_2016_google2.iloc[i]['retweet_num']+1
  res=''
  if analysis.sentiment.polarity > 0:
      count_pos+=likes+retweets
      res= 'positive'
  elif analysis.sentiment.polarity == 0:
      count_nt+=likes+retweets
      res= 'neutral'
  else:
      count_neg+=likes+retweets
      res= 'negative'
  analysis_list.append(res)

print('positive tweets ', count_pos)
print('negative tweets ', count_neg)
print('neutral tweets ', count_nt)
if count_pos> count_neg:
   print("overall sentiment: positive")
else:
  print("overall sentiment: negative")

df_2016_google2['sentiment']= analysis_list
df_2016_google2.to_csv('drive/MyDrive/ALDA data/2016_google2.csv', encoding='utf-8', index=False)


#  ALPHABET-C 2017

# In[ ]:


df_2017_google2 = df_merged[df_merged['ticker_symbol']=='GOOGL']
df_2017_google2 = df_2017_google2.loc[df_merged['year']==2015]
df_2017_google2= df_2017_google2[:5000]

from textblob import TextBlob

analysis_list= []
count_pos= 0
count_neg= 0
count_nt= 0
for i in range(len(df_2017_google2)):
  # print(i)
  analysis = TextBlob(df_2017_google2.iloc[i]['body'])
  likes= df_2017_google2.iloc[i]['like_num']+1
  retweets=df_2017_google2.iloc[i]['retweet_num']+1
  res=''
  if analysis.sentiment.polarity > 0:
      count_pos+=likes+retweets
      res= 'positive'
  elif analysis.sentiment.polarity == 0:
      count_nt+=likes+retweets
      res= 'neutral'
  else:
      count_neg+=likes+retweets
      res= 'negative'
  analysis_list.append(res)

print('positive tweets ', count_pos)
print('negative tweets ', count_neg)
print('neutral tweets ', count_nt)
if count_pos> count_neg:
   print("overall sentiment: positive")
else:
  print("overall sentiment: negative")

df_2017_google2['sentiment']= analysis_list
df_2017_google2.to_csv('drive/MyDrive/ALDA data/2017_google2.csv', encoding='utf-8', index=False)


#  ALPHABET-C 2018

# In[ ]:


df_2018_google2 = df_merged[df_merged['ticker_symbol']=='GOOGL']
df_2018_google2 = df_2018_google2.loc[df_merged['year']==2018]
df_2018_google2= df_2018_google2[:5000]

from textblob import TextBlob

analysis_list= []
count_pos= 0
count_neg= 0
count_nt= 0
for i in range(len(df_2018_google2)):
  # print(i)
  analysis = TextBlob(df_2018_google2.iloc[i]['body'])
  likes= df_2018_google2.iloc[i]['like_num']+1
  retweets=df_2018_google2.iloc[i]['retweet_num']+1
  res=''
  if analysis.sentiment.polarity > 0:
      count_pos+=likes+retweets
      res= 'positive'
  elif analysis.sentiment.polarity == 0:
      count_nt+=likes+retweets
      res= 'neutral'
  else:
      count_neg+=likes+retweets
      res= 'negative'
  analysis_list.append(res)

print('positive tweets ', count_pos)
print('negative tweets ', count_neg)
print('neutral tweets ', count_nt)
if count_pos> count_neg:
   print("overall sentiment: positive")
else:
  print("overall sentiment: negative")

df_2018_google2['sentiment']= analysis_list
df_2018_google2.to_csv('drive/MyDrive/ALDA data/2018_google2.csv', encoding='utf-8', index=False)


# In[ ]:




