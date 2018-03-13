# -*- coding: utf-8 -*-
"""
Created on Sun Mar 11 19:35:17 2018

@author: Brooke
"""

# HW8 Scripts
import pandas as pd
import numpy as np

from sklearn.cluster import KMeans, AgglomerativeClustering

from sklearn import tree, svm, metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_predict, cross_val_score, KFold
from sklearn.datasets import load_digits
from sklearn.preprocessing import scale
from sklearn.datasets import fetch_20newsgroups
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import silhouette_samples, silhouette_score

import nltk
from nltk.corpus import stopwords

import re

import matplotlib.pyplot as plt
import matplotlib.cm as cm
#%matplotlib inline
plt.rcParams['figure.figsize'] = (10, 6)
plt.style.use('ggplot')

from matplotlib.colors import ListedColormap
cmap = ListedColormap(["#e41a1c","#984ea3","#a65628","#377eb8"])
#%% Part 1 #################################################

# Task 1.1

# Your code here
crimedf = pd.read_csv('USarrests.csv')
crimedf.head()

crimedf.describe()

crimedf.dtypes

crime_corr = crimedf.corr()
crime_corr

ind = np.arange(len(list(crime_corr))) + 0.5
labels = list(crime_corr)[1:]
plt.pcolor(crime_corr,vmin=-1,vmax=1)
plt.xticks(ind,list(crime_corr),rotation=90)
plt.yticks(ind,list(crime_corr))
plt.colorbar()
plt.title('Heat Map of Coefficients')

pd.plotting.scatter_matrix(crime_corr, figsize=(10, 10), diagonal='kde')
plt.show()

# only showing like 14 points, not all 50?  ??????????????????????????

#%% # Task 1.2

scaledX = scale(crimedf[['Murder','Assault','UrbanPop','Rape']])
#check
print('mean check:',scaledX.mean(axis=0))
print('std dev check:',scaledX.std(axis=0))

#plt.scatter(scaleddf['Murder'],crimedf['Unnamed: 0'])

#data_pred = KMeans(n_clusters=4,n_init=100).fit_predict(scaledX)
data_pred = KMeans(n_clusters=4,).fit_predict(scaledX)
plt.scatter(scaledX[:, 0], scaledX[:, 1], c=data_pred,  marker="o") #, cmap=cmap);

# Which states belong to which clusters?  ??????????????????????????

#%% determining best k: measuring intra-cluster distances

# clustering for k = 1 to k = 10
ks = range(1,50)
scores = []

for k in ks:
    model = KMeans(n_clusters=k)
    model.fit_predict(scaledX)
    scores.append(-model.score(scaledX))

plt.plot(ks, scores)
plt.ylabel('total inntra-cluster distance')
plt.xlabel('k')
plt.show()

#%% determining best k: silhouette analysis

range_n_clusters = [2, 3, 4, 5, 6]


for n_clusters in range_n_clusters:
    # Create a subplot with 1 row and 2 columns
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)

    ax1.set_xlim([-0.1, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim([0, len(scaledX) + (n_clusters + 1) * 10])

    # Initialize the clusterer with n_clusters value and a random generator
    # seed of 10 for reproducibility.
    clusterer = KMeans(n_clusters=n_clusters, random_state=10)
    cluster_labels = clusterer.fit_predict(scaledX)

    silhouette_avg = silhouette_score(scaledX, cluster_labels)
    print("For n_clusters =", n_clusters,
          "The average silhouette_score is :", silhouette_avg)

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(scaledX, cluster_labels)

    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = \
            sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
#    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    # 2nd Plot showing the actual clusters formed
    colors = cm.spectral(cluster_labels.astype(float) / n_clusters)
    ax2.scatter(scaledX[:, 0], scaledX[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                c=colors, edgecolor='k')

    # Labeling the clusters
    centers = clusterer.cluster_centers_
    # Draw white circles at cluster centers
    ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
                c="white", alpha=1, s=200, edgecolor='k')

    for i, c in enumerate(centers):
        ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                    s=50, edgecolor='k')

    ax2.set_title("The visualization of the clustered data.")
    ax2.set_xlabel("Feature space for the 1st feature")
    ax2.set_ylabel("Feature space for the 2nd feature")

    plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                  "with n_clusters = %d" % n_clusters),
                 fontsize=14, fontweight='bold')

    plt.show()
    
# so best n_clusters = 2?
# For n_clusters = 2 The average silhouette_score is : 0.408489032622

#%%