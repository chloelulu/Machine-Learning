
# coding: utf-8

# # 1. Data exploration

# In[180]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations_with_replacement


# In[181]:


names = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model_year', 'origin', 'car_name']


# In[182]:


data = pd.read_csv('auto-mpg.data', sep='\s+',header=None, names = names, na_values = ['?'])
# Data exploration
data.head()
# types of the attributes or features
data.dtypes


# In[183]:


# missing value
data.isnull().sum()


# Summary of data exploration: 
# The file is csv format with Comma-Separated Values. 
# There is no index for each column, therefore during the data import, the data was manually labelled based on the description in name file. 
# For all attirbutes, cylinders, model year, origin are integer format, car name is character format, while the rest are float format.
# There are some missing values in the attibute horsepower.

# # 2. Data preparation

# (1)Compute statistics for the attributes

# In[184]:


data.describe()


# #(2)Decide on an imputation strategy for missing or incorrect data. Document your reasons.
# To decide how to handle missing data, it is helpful to know why they are missing. Based on the raw data, it seems that the data was missing randomly. Therefore, I decide to impute the missing value with mean.
# So if the data are missing completely at random, the estimate of the mean remains unbiased. Plus, by imputing the mean, you are able to keep your sample size up to the full sample size.

# In[185]:


# (3)Apply your imputation strategy.
# fill missing values with mean column values
data.fillna(data.mean(), inplace=True)
# count the number of NaN values in each column
print(data.isnull().sum())
data.describe()


# # 3.Implement the algorithm
# # Linear regression

# In[186]:


def MSE(w,x,y):
    r = x.dot(w)-y    
    return r.dot(r)/len(x)

def LM(X,Y):
    pos_inv = np.linalg.pinv(X)
    w = np.dot(pos_inv,Y)
    return w


# In[187]:


# Target - Features Split
y = np.array(data['mpg'])
X = np.array(data.drop(['mpg','car_name'], axis=1))
# Train - Test -Val Split
from sklearn.model_selection import train_test_split
# First to split to train, test and then split train again into validation and train. Something like this:
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=1)

# add one as the first column
def add_one(x):
    one = np.ones([x.shape[0],1])
    x = np.concatenate((one,x),axis=1)
    return x


# In[188]:


mse = []
for i in range(0,50):
    # First to split to train, test and then split train again into validation and train. Something like this:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2)
    X_train = add_one(X_train)
    lm_train = LM(X_train, y_train)
    mse_train = MSE(lm_train, X_train, y_train)
    X_val = add_one(X_val)
    mse_val = MSE(lm_train, X_val, y_val)
    mse.append([mse_train,mse_val])
mse_df = pd.DataFrame(mse,columns=['train_mse','validate_mse'])
mse_df  


# In[189]:


mse_df.mean() 


# In[190]:


# Standardization
from sklearn import preprocessing
# Target - Features Split
y = np.array(data['mpg'])
X = np.array(data.drop(['mpg','car_name'], axis=1))

def std(x):
    scaler_X = preprocessing.StandardScaler().fit(x)
    x = scaler_X.transform(x)
    return x


mse_sd = []
for i in range(0,50):
    # First to split to train, test and then split train again into validation and train. Something like this:
    X_train_scaled, X_test_scaled, y_train, y_test = train_test_split(X, y, test_size=0.2)
    X_train_scaled, X_val_scaled, y_train, y_val = train_test_split(X_train_scaled, y_train, test_size=0.2)
    
    X_train_scaled = std(X_train_scaled)
    X_val_scaled = std(X_val_scaled)
    
    X_train_scaled = add_one(X_train_scaled)
    X_val_scaled = add_one(X_val_scaled)
    
    lm_train_scaled = LM(X_train_scaled, y_train)
    mse_train_scaled = MSE(lm_train_scaled, X_train_scaled, y_train)

    mse_val_scaled = MSE(lm_train_scaled, X_val_scaled, y_val)
    mse_sd.append([mse_train_scaled,mse_val_scaled])

mse_df_scaled = pd.DataFrame(mse_sd,columns=['train_mse_sd','validate_mse_sd'])
mse_df_scaled


# In[191]:


mse_df_scaled.mean()


# # For multivariate polynomial regression

# In[192]:


# For non-standarized
def icom(features_number,degree):
    x = [combinations_with_replacement(range(features_number), i) for i in range(0, degree + 1)]
    y = [item for sublist in x for item in sublist]
    return y

def polynomial(X, degree):
    sample_number, features_number = np.shape(X)
        
    join = icom(features_number,degree)
    output_features_number = len(join)
    X1 = np.empty((sample_number, output_features_number))

    for i, x in enumerate(join):  
        X1[:, i] = np.prod(X[:, x], axis=1)
        
    return X1   

def pm(X, Y, degree):
    X1 = polynomial(X, degree)
    pinv = np.linalg.pinv(X1)
    w = pinv.dot(Y)
    return w


# In[193]:


# Non-Standardization
from sklearn import preprocessing
# Target - Features Split
y = np.array(data['mpg'])
X = np.array(data.drop(['mpg','car_name'], axis=1))

for n in range(2,5):
    mse = []
    for i in range(0,50):
        # First to split to train, test and then split train again into validation and train. Something like this:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2)

        p_train = polynomial(X_train,n)
        p_val = polynomial(X_val,n)

        lm_train = pm(X_train, y_train,n)
        mse_train = MSE(lm_train, p_train, y_train)

        mse_val = MSE(lm_train, p_val, y_val)
        mse.append([mse_train,mse_val])

    mse_df = pd.DataFrame(mse,columns=['train_mse_sd','validate_mse_sd'])
    print('Degree:' + str(n))
    print(mse_df.mean())


# In[194]:


# Standardization
from sklearn import preprocessing
# Target - Features Split
y = np.array(data['mpg'])
X = np.array(data.drop(['mpg','car_name'], axis=1))

def std(x):
    scaler_X = preprocessing.StandardScaler().fit(x)
    x = scaler_X.transform(x)
    return x

for n in range(2,5):
    mse_sd = []
    for i in range(0,50):
        # First to split to train, test and then split train again into validation and train. Something like this:
        X_train_scaled, X_test_scaled, y_train, y_test = train_test_split(X, y, test_size=0.2)
        X_train_scaled, X_val_scaled, y_train, y_val = train_test_split(X_train_scaled, y_train, test_size=0.2)

        X_train_scaled = std(X_train_scaled)
        X_val_scaled = std(X_val_scaled)

        p_train = polynomial(X_train_scaled,n)
        p_val = polynomial(X_val_scaled,n)

        lm_train_scaled = pm(X_train_scaled, y_train,n)
        mse_train_scaled = MSE(lm_train_scaled, p_train, y_train)

        mse_val_scaled = MSE(lm_train_scaled, p_val, y_val)
        mse_sd.append([mse_train_scaled,mse_val_scaled])

    mse_df_scaled = pd.DataFrame(mse_sd,columns=['train_mse_sd','validate_mse_sd'])
    print('Degree:' + str(n))
    print(mse_df_scaled.mean())

    

