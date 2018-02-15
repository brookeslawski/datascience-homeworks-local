# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 09:42:45 2018

@author: Brooke
"""

# HW5 Scripts

# Task 1
# imports and setup 
import pandas as pd
import scipy as sc
import numpy as np

import statsmodels.formula.api as sm

import matplotlib.pyplot as plt 
plt.style.use('ggplot')
#%matplotlib inline  
plt.rcParams['figure.figsize'] = (10, 6) 
# your code goes here
data1 = pd.read_csv('train1.csv',index_col=0)
data2 = pd.read_csv('train2.csv',index_col=0)
data = pd.concat([data1,data2],axis=0)#top/bottom
data.head()
print('data1 shape: ' + str(data1.shape))
print('data2 shape: ' + str(data2.shape))
print('data shape: ' + str(data.shape))