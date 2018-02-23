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
# for i in np.arange(1,11):
    
# download the fist ten pages

for i in np.arange(1,11):

    url = "https://github.com/search?o=desc&p="+str(i)+"&q=stars%3A%3E1&s=stars&type=Repositories"

    with urllib.request.urlopen(url) as response:
        html = response.read()
        html = html.decode('utf-8')
        
    time.sleep(10) #sleeps after every 5 request

    #A simple solution is to just use ‘response.text’ to get html returned by requests.get(url) directly. This doesn't work on Windows.
    # html = requests.get(url)
    # html = html.text

    # # save the file
    filename = "repos"+str(i)+".html"
    with open(filename, 'w') as new_file:
        new_file.write(str(html.replace('\n','').encode('utf-8')))
    
    # create one massive soup
    if i == 1:
        git_soup = BeautifulSoup(html, 'html.parser') # initialize git_soup w/ first page
    else:
        cup_soup = BeautifulSoup(html, 'html.parser') # soup for each page
        for element in cup_soup:
            git_soup.body.append(element) # massive soup that contains soups of all pages

# massive soup that contains soups of all pages was created in for loop above
git_soup

# %% Task 1.3

name_list = [] # initialize lists
url_list = [] 
cont_list = [] 
lang_list = [] 
stars_list = []
issues_list = []
forks_list = []
rmlength_list = []

repos = git_soup.find_all("div",class_="col-8 pr-3")
repos_lang = git_soup.find_all("div",class_="d-table-cell col-2 text-gray pt-2")
url = "https://github.com"

for i in range(100):
    name_list.append(repos[i].find("a",class_="v-align-middle").text)
    print(repos[i].find("a",class_="v-align-middle").text)
    url_tail = repos[i].find("a").get("href")
    repo_url = url + url_tail
    url_list.append(repo_url)
    with urllib.request.urlopen(repo_url) as response:
        repo_html = response.read()
        repo_html = repo_html.decode('utf-8')
    # if i%5==0:
    #     time.sleep(5) #sleeps after every 5 request
    #     # adjust frequency or time for sleeping if have issues
    with open('repo.html', 'w') as new_file:
        new_file.write(str(repo_html.replace('\n','').encode('utf-8')))
    repo_soup = BeautifulSoup(repo_html, 'html.parser')
    cont_list.append(repo_soup.find("a", href= lambda x : x and "contributors" in x).text.replace('\n','').strip('contributors').strip())
    lang_list.append(repos_lang[i].text.strip())
    stars_list.append(repo_soup.find("a",class_="social-count js-social-count").text.strip())
    issues_list.append(repo_soup.find("span",class_="Counter").text)
    forks_list.append(repo_soup.find("a", href= lambda x : x and "network" in x).text.strip())
    rmlength_list.append(len(repo_soup.find("div",id="readme").text.replace("\n","")))
    
# print('names:',name_list,'\n')
# print('URLs:',url_list,'\n')
# print('cont_list:',cont_list,'\n')
# print('lang_list:',lang_list,'\n')
# print('# of stars:',stars_list,'\n')
# print('# of issues:',issues_list,'\n')
# print('# of forks:',forks_list,'\n')
# print('RM lengths:',rmlength_list,'\n')

# %%

# Convert list into DataFrame
reposDF = pd.DataFrame({"Repository Names":name_list,
                       "URLs":url_list,
                       "Contributors":cont_list,
                       "Language":lang_list,
                       "Stars":stars_list,
                       "Issues":issues_list,
                       "Forks":forks_list,
                       "ReadMeLength":rmlength_list})

reposDF

# Save dataframe to file project_info.csv
reposDF.to_csv('project_info.csv', encoding='utf-8')

# %% Task 2

# this loads the data from the project_info.csv file 
project_info1 = pd.read_csv('project_info.csv')
# get rid of index column:
project_info = project_info1[['Repository Names','Contributors','Forks','Issues','Language','ReadMeLength','Stars','URLs']]
project_info = project_info.set_index('Repository Names')
project_info.head()


# %% Task 2.1

print('Pre- data types: \n',project_info.info(),'\n')
project_info['Issues'] = project_info['Issues'].astype(str).str.replace(',','').astype(int)
project_info['Forks'] = project_info['Forks'].astype(str).str.replace(',','').astype(int)
project_info['Stars'] = project_info['Stars'].astype(str).str.replace(',','').astype(int)
project_info.loc['torvalds/linux','Contributors'] = 15000
project_info['Contributors'] = project_info['Contributors'].astype(str).str.replace(',','').astype(int)
print('Post- data types: \n',project_info.info())

project_info.head(12)

# %% Task 2.2

project_info.describe()

# what are these two outliers? see explanation for this code in the last "Your Interpretation" section:
# project_info.loc[project_info['Stars'] == 291631] #which repository has the max number of stars?
# project_info = project_info[project_info['Stars'] != 291631] #mask to get rid of outliers
# project_info = project_info[project_info['Contributors'] != 15000] #mask to get rid of outliers
# project_info.shape

pi_corr = project_info.corr()
pi_corr

ind = np.arange(len(list(pi_corr))) + 0.5
labels = list(pi_corr)[1:]
plt.pcolor(pi_corr,vmin=-1,vmax=1)
plt.xticks(ind,list(pi_corr),rotation=90)
plt.yticks(ind,list(pi_corr))
plt.colorbar()
plt.title('Heat Map of Coefficients')

pd.plotting.scatter_matrix(project_info, figsize=(10, 10), diagonal='kde')
plt.show()

# %% Task 2.3
pi_ols = sm.ols(formula="Stars ~ Forks + Contributors + Issues + ReadMeLength", data=project_info).fit()
print('Proposed model \n',pi_ols.summary(),'\n')

# pi_ols2 = sm.ols(formula="Stars ~ Forks", data=project_info).fit() # yields an r-squared value > 0.5 after two outlier repositories are removed from data
# print('Improved model \n',pi_ols2.summary())

pi_ols2 = sm.ols(formula="Stars ~ Forks + Issues + Forks*Issues", data=project_info).fit()
print('Improved model \n',pi_ols2.summary())

