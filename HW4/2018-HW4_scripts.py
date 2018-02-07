# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 13:16:27 2018

@author: Brooke
"""

# HW4 scripts

import pandas as pd
import numpy as np
import scipy as sc
from scipy.stats import norm

import matplotlib.pyplot as plt
#%matplotlib inline
plt.rcParams['figure.figsize'] = (10, 6)
plt.style.use('ggplot')

# Task 1.2
aqd_csv = pd.read_csv("2017airqualitydata.csv")
aqd = pd.DataFrame(aqd_csv)
aqd.head()

print("Number of rows loaded = " + str(aqd.shape[0]) + ", Number of columns loaded = " + str(aqd.shape[1]) + "\n")
aqd.info()

#%%Task 1.3
aqd_trunc = aqd[["Date","POC","Daily Mean PM2.5 Concentration","UNITS","DAILY_AQI_VALUE"]]
aqd_date = aqd_trunc.groupby("Date")
fig1 = plt.figure()
aqd_date.mean()[["Daily Mean PM2.5 Concentration", "DAILY_AQI_VALUE"]].plot() 

POC1 = aqd[aqd["POC"]==1].groupby("Date").mean().sort_index()
POC4 = aqd[aqd["POC"]==4].groupby("Date").mean().sort_index()
POC5 = aqd[aqd["POC"]==5].groupby("Date").mean().sort_index()

fig2 = plt.figure()
plt.plot(POC1[["Daily Mean PM2.5 Concentration"]], label="POC 1")
plt.plot(POC4[["Daily Mean PM2.5 Concentration"]], label="POC 4")
plt.plot(POC5[["Daily Mean PM2.5 Concentration"]], label="POC 5")
plt.legend()
plt.title("Daily Mean PM2.5 Concentration by POC")
plt.show()

fig3 = plt.figure()
plt.plot(POC1[["DAILY_AQI_VALUE"]], label="POC 1")
plt.plot(POC4[["DAILY_AQI_VALUE"]], label="POC 4")
plt.plot(POC5[["DAILY_AQI_VALUE"]], label="POC 5")
plt.legend()
plt.title("Daily AQI Value by POC")
plt.show()

#%% Task 1.4
fig4 = plt.figure()
aqi = aqd_date.mean()[["DAILY_AQI_VALUE"]]
mvg_avg = np.round(aqi["DAILY_AQI_VALUE"].rolling(window = 20, center = False).mean(), 2)
plt.plot(aqi, label="AQI Daily Values")
plt.plot(mvg_avg, label="AQI Moving Average")
plt.legend()
plt.title("AQI Daily and Mean Values")
plt.show()

#%%