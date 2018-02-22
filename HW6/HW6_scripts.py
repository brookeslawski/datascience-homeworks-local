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
for i in np.arange(1,11):
    
url = "https://github.com/search?o=desc&p=1&q=stars%3A%3E1&s=stars&type=Repositories"


with urllib.request.urlopen(url) as response:
    html = response.read()
    html = html.decode('utf-8')

#A simple solution is to just use ‘response.text’ to get html returned by requests.get(url) directly. This doesn't work on Windows (only Mac and Linux)
# html = requests.get(url)
# html = html.text

# # save the file
with open('git_repositories.html', 'w') as new_file:
    new_file.write(str(html.replace('\n','').encode('utf-8')))

git_soup = BeautifulSoup(html, 'html.parser')
git_soup = BeautifulSoup(html, 'html.parser')


name_list = []
url_list = []
cont_list = []

repos = git_soup.find_all("div",class_="col-8 pr-3")
url = "https://github.com"

i = 0
#for i in range(10):
    name_list.append(repos[i].find("a",class_="v-align-middle").text)
    url_tail = repos[i].find("a").get("href")
    repo_url = url + url_tail
    url_list.append(repo_url)
    with urllib.request.urlopen(repo_url) as response:
        repo_html = response.read()
        repo_html = repo_html.decode('utf-8')
    with open('repo.html', 'w') as new_file:
        new_file.write(str(repo_html.replace('\n','').encode('utf-8')))
    repo_soup = BeautifulSoup(repo_html, 'html.parser')
    cont_list.append(repo_soup.find("a", href= lambda x : x and "contributors" in x).text)
    
    
    
    
    #num_cont_list.append(repos[i].find("a").get("href"))

print('names:',name_list,'\n')
print('URLs:',url_list,'\n')
print('cont_list:',cont_list,'\n')