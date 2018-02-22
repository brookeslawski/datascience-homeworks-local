# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 11:03:35 2018

@author: Brooke
"""

# HW 6 scripts

# hw: write a for loop to look at first 10 pages:
# for range 0 to 10, change l dynamically
# url has page number

# imports and setup 
from bs4 import BeautifulSoup
# you can use either of these libraries to get html from a website
import requests
import urllib.request


import pandas as pd
import scipy as sc
import numpy as np

import statsmodels.formula.api as sm

import matplotlib.pyplot as plt 
plt.style.use('ggplot')
#%matplotlib inline  
plt.rcParams['figure.figsize'] = (10, 6) 

# %% Task 1.2
## Your code goes here

# download the fist ten pages
url = "https://github.com/search?o=desc&p=1&q=stars%3A%3E1&s=stars&type=Repositories"

# with urllib.request.urlopen(url) as response:
#     html = response.read()
#     html = html.decode('utf-8')

#A simple solution is to just use ‘response.text’ to get html returned by requests.get(url) directly.
    
requests.get(url)
html = response.text()

# save the file
with open('git_repositories.html', 'w') as new_file:
    new_file.write(html)

# here it's already a local operation
git_soup = BeautifulSoup(html, 'html.parser')