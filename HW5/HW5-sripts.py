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

housedata=housedata.drop(['ListAgt','AirType','Amenities','Basemnt','Window','Access','DirPre','Zoning','P2Bed',
 'P2BthFull',
 'P2BthHalf',
 'P2BthTq',
 'P2Fire',
 'P2Rent',
 'P2Sqf',
 'P2FamDen',
 'P2Kitch',
 'P2Bar',
 'P2Formal',
 'P2SemiForm',
 'P2Laundry',
 'P3Bed',
 'P3BthFull',
 'P3BthHalf',
 'P3BthTq',
 'P3Fire',
 'P3Rent',
 'P3Sqf',
 'P3FamDen',
 'P3Kitch',
 'P3Bar',
 'P3Formal',
 'P3SemiForm',
 'P3Laundry',
 'P4Bed',
 'P4BthFull',
 'P4BthHalf',
 'P4BthTq',
 'P4Fire',
 'P4Rent',
 'P4Sqf',
 'P4FamDen',
 'P4Kitch',
 'P4Bar',
 'P4Formal',
 'P4SemiForm',
 'P4Laundry',
 'P5Bed',
 'P5BthFull',
 'P5BthHalf',
 'P5BthTq',
 'P5Fire',
 'P5Rent',
 'P5Sqf',
 'P5FamDen',
 'P5Kitch',
 'P5Bar',
 'P5Formal',
 'P5SemiForm',
 'DirPost',
 'SldOfcID',
 'SlrPaidCns',
 'NumDispose',
 'NumOvRng',
 'Owner',
 'TaxID',
 'Sewer',
 'Show',
 'P5Laundry',], axis=1) #removing categorical variables with too many unique options to regress

#housedata=truncdata.drop(['AdditInfo','Area','BackDim','TotWinEvp':'WinEvp4'], axis=1)
print(housedata.shape)
housedata.dtypes
housedata['TotSqf'] = housedata['TotSqf'].astype(str).str.replace(",","").astype(int)
housedata.dtypes

#%% If there are any categorical values you're interested in, then you should convert them to numerical values as in Lecture 11. In particular, convert 'TotSqf' to an integer and add a column titled Prop_Type_num that is

housedata["Prop_Type_num"] = housedata["PropType"].map({'Condo':0, 'Townhouse':0, 'Single Family':1}) #convert categorical var to numbers
housedata["SchSenior_num"] = housedata["SchSenior"].map({'West':0, 'East':1}) #convert categorical var to numbers
print("check townhouse: ") 
housedata[housedata['ListNo'] == 1397571]

#%% Remove the listings with erroneous 'Longitude' (one has Longitude = 0) and 'Taxes' values (two have unreasonably large values).
print(housedata.shape)
housedata['Longitude'].describe()
housedata = housedata[housedata['Longitude'] != 0] #mask to get ride of Longitude = 0 row
housedata['Taxes'].describe()
housedata = housedata[housedata['Taxes'] < 95000]
print(housedata.shape) #should remove 3 rows

# %%

# Task 3.2 make a bar chart
# bar chart of property type
housedata["PropType"].value_counts().plot(kind='bar',title="Property Type Breakdown")


# %%
# Task 3.3

#reduced variables to test
hdf = housedata[['Acres', 'Deck', 'GaragCap', 'Latitude', 'Longitude', 'LstPrice', 'Patio', 'PkgSpacs', 'PropType', 'SoldPrice', 'Taxes', 'TotBed', 'TotBth', 'TotSqf', 'PckSpacs', 'YearBlt','Prop_Type_num']]

hdf_corr = hdf.corr()
print(hdf_corr)

ind = np.arange(len(list(hdf_corr))) + 0.5
labels = list(hdf_corr)[1:]
plt.pcolor(hdf_corr,vmin=-1,vmax=1)
plt.xticks(ind,list(hdf_corr),rotation=90)
plt.yticks(ind,list(hdf_corr))
plt.colorbar()
plt.title('Heat Map of Coefficients')

# heatmap


# %% Task 3.4
subset = hdf[['Acres', 'LstPrice', 'SoldPrice', 'Taxes', 'TotBed', 'TotBth', 'TotSqf', 'YearBlt']]

pd.plotting.scatter_matrix(subset, figsize=(10, 10), diagonal='kde')
plt.show()

# %%
# Task 4
hdf.plot.scatter(x="Longitude", y="Latitude")

# %% Task 5

sp_tsf = sm.ols(formula="SoldPrice ~ TotSqf", data=hdf).fit()
sp_tsf.summary()

sp_tb = sm.ols(formula="SoldPrice ~ TotBed", data=hdf).fit()
sp_tb.summary()

sp_yb = sm.ols(formula="SoldPrice ~ YearBlt", data=hdf).fit()
sp_yb.summary()

sp_ls = sm.ols(formula="SoldPrice ~ LstPrice", data=hdf).fit()
sp_ls.summary()

# %% Task 5 still

plt.scatter(x=hdf['LstPrice'],y=hdf['SoldPrice'],c='k',label='TV')
plt.plot(hdf['LstPrice'],sp_ls.predict(),'k',color='blue',linewidth=3)
# ad_TV_ols.predict displays predicted y values for those values of x

plt.xlabel('List Price ($)')
plt.ylabel('Sold Price ($)')
plt.show()

# %% Task 6

#Variables: Acres, Deck, GaragCap, Latitude, Longitude, Patio, PkgSpacs, Taxes, TotBed, TotBth, TotSqf, YearBlt

#sp_all_ols = sm.ols(formula="SoldPrice ~ Acres + Deck + GaragCap + Latitude + Longitude + Patio + PkgSpacs + TotBed + TotBth + TotSqf + YearBlt", data=hdf).fit()
# Parking Spaces, Patio, Deck, TotBed, and YearBlt have high p-values so got rid of them

sp_all_ols = sm.ols(formula="SoldPrice ~ Acres + GaragCap + Latitude + Longitude + TotBth + TotSqf", data=hdf).fit()
#Latitude has high p-value so getting rid of it

sp_all_ols = sm.ols(formula="SoldPrice ~ Acres + GaragCap + Longitude + TotBth + TotSqf", data=hdf).fit()
# but then R-squred val went down

sp_all_ols.summary()

# %% Task 7

sp_ptn = sm.ols(formula="SoldPrice ~ Prop_Type_num", data=hdf).fit()
sp_ptn.summary()
# r-squared is really low?  how is this good at predicting?

# %%

sp_ptn_tsf = sm.ols(formula="SoldPrice ~ Prop_Type_num + TotSqf", data=hdf).fit()
sp_ptn_tsf.summary()

# r-squared is better, but p-val is terrible
# %% TotSqf vs. SoldPrice

# hdf.plot.scatter(x="TotSqf", y="SoldPrice")
plt.scatter(x=hdf["TotSqf"], y=hdf["SoldPrice"])
plt.scatter(x=hdf[hdf["PropType"]=="Single Family"],y=hdf["SoldPrice"],  color='red',label='Single Family')
plt.scatter(x=hdf[hdf["PropType"]=="Condo"],y=hdf["SoldPrice"],  color='blue',label='Condo')

plt.legend()
plt.xlabel('Total Square Ft')
plt.ylabel('Sold Price ($)')
plt.show()
#plt.scatter(X_test[:,0], SoldPrice,  color='black',label='Condo')
#plt.scatter(X_test[:,0], SoldPrice,  color='black',label='Condo')

#%%

# questions:

# how to view all columns that have 3 or fewer unique values
# how to view all column names (keeps truncating with ... b/c too many)
# is there a difference between ' and "
