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
# your code goes here
data1 = pd.read_csv('train1.csv')
data2 = pd.read_csv('train2.csv')
data = pd.concat([data1,data2],axis=0)#top/bottom
data.head()
print('data1 shape: ' + str(data1.shape))
print('data2 shape: ' + str(data2.shape))
print('data shape: ' + str(data.shape)) 

#%% Task 2

truncdata = data[data["LstPrice"] >= 200000]
truncdata = truncdata[truncdata["LstPrice"] <= 1000000]
print(truncdata.shape)
housedata=truncdata.drop(['AdditInfo','Area','BackDim','City','ListCoAgt','CompBac','CompDays','CompSac','ContPh1','ContPh2','Contact','County','EntrdBy','HouseNbr','NumDish','NumRefg','OpenHseDt','PublicID','RMPriceLow','ReinstDt','Remarks','SchDist','SlAgentPub','StatCode','State','StrType','SubAgncy','TimeClause','TotWinEvp','UndrCnst','UnitNbr','Water','WinEle1','WinEle2','WinEle3','WinEle4','WinEvp1','WinEvp2','WinEvp3','WinEvp4','WithDrDt'], axis=1)
#housedata=truncdata.drop(['AdditInfo','Area','BackDim','TotWinEvp':'WinEvp4'], axis=1)
print(housedata.shape)
housedata.dtypes
housedata['TotSqf'] = housedata['TotSqf'].astype(str).str.replace(",","").astype(int)
housedata.dtypes

#%% If there are any categorical values you're interested in, then you should convert them to numerical values as in Lecture 11. In particular, convert 'TotSqf' to an integer and add a column titled Prop_Type_num that is

housedata["Prop_Type_num"] = housedata["PropType"].map({'Condo':0, 'Townhouse':0, 'Single Family':1}) #convert categorical var to numbers
housedata["SchSenior_num"] = housedata["SchSenior"].map({'West':0, 'East':0}) #convert categorical var to numbers
print("check townhouse: ") 
housedata[housedata['ListNo'] == 1397571]

#%% Remove the listings with erroneous 'Longitude' (one has Longitude = 0) and 'Taxes' values (two have unreasonably large values).
print(housedata.shape)
housedata['Longitude'].describe()
housedata = housedata.drop(housedata.index[housedata['Longitude'].idxmax()],axis=0)
print(housedata.shape)

#%%

# questions:

# how to view all columns that have 3 or fewer unique values
# how to view all column names (keeps truncating with ... b/c too many)

