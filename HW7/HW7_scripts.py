# -*- coding: utf-8 -*-
"""
Created on Sat Mar  3 13:22:28 2018

@author: Brooke
"""

#HW 7 Scripts

import pandas as pd
import numpy as np

from sklearn import tree, svm, metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_predict, cross_val_score, KFold
from sklearn.datasets import load_digits
from sklearn.preprocessing import scale

import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (10, 6)
plt.style.use('ggplot')

# Part 1

digits = load_digits()
X = scale(digits.data)
y = digits.target
print(type(X))

n_samples, n_features = X.shape
n_digits = len(np.unique(digits.target))
print("n_digits: %d, n_samples %d, n_features %d" % (n_digits, n_samples, n_features))

# this is what one digit (a zero) looks like
print("===\nThe raw data")
print(digits.images[0])
print("===\nThe scaled data")
print(X[0])
print("===\nThe digit")
print(digits.target[0])

plt.figure(figsize= (10, 10))    
for ii in np.arange(25):
    plt.subplot(5, 5, ii+1)
    plt.imshow(np.reshape(X[ii,:],(8,8)), cmap='Greys',interpolation='nearest')
    plt.axis('off')
plt.show()

#%% Task 1.1

# your solution goes here
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, test_size=0.3) # test_size=0.8 doesn't make sense

dig_svm = svm.SVC(kernel='rbf',C=100)  # change this one 6, 5, 4, 3, 2
dig_svm.fit(X_train, y_train)

y_pred = dig_svm.predict(X_test)
print('Confusion Matrix')
print(metrics.confusion_matrix(y_true = y_test, y_pred = y_pred))
print('Accuracy = ', metrics.accuracy_score(y_true = y_test, y_pred = y_pred))

#%% Task 1.1.4 print misclassified digits as images
inds = np.intersect1d(np.where(y_test==3),np.where(y_pred==2))
print(inds)

for ii,ind in enumerate(inds):
    plt.subplot(1, 2, ii+1)
    plt.imshow(np.reshape(X_test[ind,:],(8,8)), cmap='Greys',interpolation='nearest')
    plt.axis('off')
    
plt.show()       

    
#%% Task 1.1.5 
dig_svm.get_params()

Cs = np.linspace(1,500,100)
Accuracies = np.zeros(Cs.shape[0])
for i,C in enumerate(Cs): # what is enumerate?
    dig_svm = svm.SVC(kernel='rbf', C = C)
    scores = cross_val_score(estimator = dig_svm, X = X, y = y, cv=5, scoring='accuracy')     # cv = 5 ??
    Accuracies[i]  = scores.mean()
    
#%%
        
plt.plot(Cs,Accuracies)
plt.title('Evaluating accuracy of C')
plt.xlabel('C value')
plt.ylabel('Accuracy')
plt.xlim(0,10) # change this to 0, 10
plt.show()

# choose any C greater than or equal to 6?
# but previously, I got 95.3% accuracy w/ C = 100.  Is this b/c the data 
# was split?

#%% Task 1.1.6

Xraw = digits.data
X_train, X_test, y_train, y_test = train_test_split(Xraw, y, random_state=1, test_size=0.3)

rawdig_svm = svm.SVC(kernel='rbf',C=10)
rawdig_svm.fit(X_train, y_train)

y_pred = rawdig_svm.predict(X_test)
print('Confusion Matrix')
print(metrics.confusion_matrix(y_true = y_test, y_pred = y_pred))
print('Accuracy = ', metrics.accuracy_score(y_true = y_test, y_pred = y_pred))

# Accuracy of 0.1182, which is terrible
# Almost all data was classified as '0' (first column)

# change test_size from 0.8 to 0.5
# Accuracy =  0.290322580645

# change test_Size from 0.5 to 0.1
# Accuracy =  0.561111111111

# is this wrong?  training the model on so much data that there is barelly enough to test it with

#%%

rawdig_svm.get_params()

Cs = np.linspace(1,500,100)
Accuracies = np.zeros(Cs.shape[0])
for i,C in enumerate(Cs): # what is enumerate?
    rawdig_svm = svm.SVC(kernel='rbf', C = C)
    scores = cross_val_score(estimator = rawdig_svm, X = Xraw, y = y, cv=5, scoring='accuracy')     # cv = 5 ??
    Accuracies[i]  = scores.mean()

#%%
    
plt.plot(Cs,Accuracies)
plt.title('Evaluating accuracy of C')
plt.xlabel('C value')
plt.ylabel('Accuracy')
plt.xlim(0,10) # change this to 0, 10
plt.show()

# %% Your interpretation

#Using the test dataset, the model's accuracy is 95.4%, meaning that 95.4% points were classified correctly. As seen in the confusion matrix, the most common mistake made by the model is found in the cell containing the value '12,' meaning that the model incorrectly classified 12 digits as '7' when they were actually a '4.' ????
#
#In Task 1.1.5, we evaluate accuracies of the SVM model and found that any C greater than or equal to 6 results in a constant maximum accuracy of 95.4%.
#
#When using a test_size of 0.8, I got an accuracy of 0.1182, which is terrible. Almost all data was classified as '0'. I then decreased the test_size from 0.8 to 0.5 to 0.1 to get an accuracy of 0.561, but this may be a questionable strategy since there is barely any data to test the algorithm on (10%).

#Evaluating several different values of C, the plot shows that any C greater than or equal to 6 should yield an accuracy of ~0.475.  This is significantly lower than the accuracy of the model on scaled data.






#%% Task 1.2.1 ##########################################################
# set up the model, k-NN classification with k = ?  
k = 10 # when k = 3, 6, not matching iterative model results?
X = scale(digits.data)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, test_size=0.3)
knn_model = KNeighborsClassifier(n_neighbors=k)
knn_model.fit(X_train, y_train)

y_pred = knn_model.predict(X_test)
print('Confusion Matrix')
print(metrics.confusion_matrix(y_true = y_test, y_pred = y_pred))
print('Accuracy = ', metrics.accuracy_score(y_true = y_test, y_pred = y_pred))

# Accuracy of 0.9124 w/ k = 10
# Most common mistake: classified 11 data points as Class 2 when really they were class 1
# why this error for k-nn and not svm?

#%% Task 1.2.4 print misclassified digits as images

inds = np.intersect1d(np.where(y_test==8),np.where(y_pred==1))
print(inds)

for ii,ind in enumerate(inds):
    plt.subplot(1, 2, ii+1)
    plt.imshow(np.reshape(X_test[ind,:],(8,8)), cmap='Greys',interpolation='nearest')
    plt.axis('off')
    
plt.show()   

inds = np.intersect1d(np.where(y_test==9),np.where(y_pred==5))
print(inds)

for ii,ind in enumerate(inds):
    plt.subplot(1, 2, ii+1)
    plt.imshow(np.reshape(X_test[ind,:],(8,8)), cmap='Greys',interpolation='nearest')
    plt.axis('off')
    
plt.show()  

#%% Task 1.2.5 

knn_model.get_params()

Ks = np.arange(1,100)
Accuracies = np.zeros(Ks.shape[0])
for i,k in enumerate(Ks):
    knn_model = KNeighborsClassifier(n_neighbors=int(k))
    scores = cross_val_score(estimator = knn_model, X = X, y = y, cv=5, scoring='accuracy')     # cv = 5 ??
    Accuracies[i]  = scores.mean()
    
#%%
    
plt.plot(Ks,Accuracies)
plt.title('Evaluating accuracy of k-values')
plt.xlabel('k value')
plt.ylabel('Accuracy')
plt.show()
    
#%%
    
plt.plot(Ks,Accuracies)
plt.title('Evaluating accuracy of k-values')
plt.xlabel('k value')
plt.ylabel('Accuracy')
plt.xlim(0,10)
plt.ylim(0.92,0.95)
plt.show()

# choose k = 6
# when k = 6, Accuracy =  0.922114047288


#%%
k = 6
#X = scale(digits.data)
#X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, test_size=0.3)
knn_model = KNeighborsClassifier(n_neighbors=k)
knn_model.fit(X_train, y_train)

y_pred = knn_model.predict(X_test)
print('Confusion Matrix')
print(metrics.confusion_matrix(y_true = y_test, y_pred = y_pred))
print('Accuracy = ', metrics.accuracy_score(y_true = y_test, y_pred = y_pred))

#%% Task 1.2.6 raw data
k = 6

Xraw = digits.data
X_train, X_test, y_train, y_test = train_test_split(Xraw, y, random_state=1, test_size=0.3)

rawdig_knn = KNeighborsClassifier(n_neighbors=k)
rawdig_knn.fit(X_train, y_train)

y_pred = rawdig_knn.predict(X_test)
print('Confusion Matrix')
print(metrics.confusion_matrix(y_true = y_test, y_pred = y_pred))
print('Accuracy = ', metrics.accuracy_score(y_true = y_test, y_pred = y_pred))

#%%

rawdig_knn.get_params()

Cs = np.linspace(1,500,100)
Accuracies = np.zeros(Cs.shape[0])
for i,C in enumerate(Cs):
    rawdig_knn = svm.SVC(kernel='rbf', C = C)
    scores = cross_val_score(estimator = rawdig_knn, X = Xraw, y = y, cv=5, scoring='accuracy')     # cv = 5 ??
    Accuracies[i]  = scores.mean()

#%%
    
plt.plot(Cs,Accuracies)
plt.title('Evaluating accuracy of C')
plt.xlabel('C value')
plt.ylabel('Accuracy')
plt.xlim(0,10) # change this to 0, 10
plt.show()






#%% Part 2 ###########################################################
# Task 2.1
newsdf = pd.read_csv('OnlineNewsPopularity/OnlineNewsPopularity.csv')
newsdf.columns = newsdf.columns.str.replace(" ","") # get rid of extra spaces in attribute names (headers)
newsdf.head()

# Task 2.2
print('News dataframe shape: ',newsdf.shape,'\n')
print('Data types: \n',newsdf.dtypes)
newsdf.tail()
newsdf.describe()

# looks like the maximum values of n_unique_tokens, n_non_stop_words, and n_non_stop_unique_tokens are not reasonable
newsdf.loc[newsdf['n_unique_tokens'] == 701.0] # which article is this? #index 31037 ukraine civilians...
# this entry also includes the maximum values of n_non_stop_words and n_non_stop_unique_tokens
newsdf = newsdf[newsdf['n_unique_tokens'] != 701.0] #remove this entry
newsdf.describe() # better

print('Cleaned news dataframe shape: ',newsdf.shape,'\n')

print('Shares Statistical info')
print(newsdf['shares'].describe())
newsdf['shares'].median()

#median number of shares = 1400

#From Task 2.1 part 3
# export predictor vars as np array
X = newsdf.drop(['url','timedelta','shares'],axis=1).as_matrix()
shares = newsdf['shares'].as_matrix()
y = [1 if x > newsdf['shares'].median() else 0 for x in shares] #binary numpy array, y, which indicates whether or not each article is popular

print('Predictor Variable Data Shape: ',X.shape)
print('Shares array length: ',len(y))

#%% Task 2.3 k-NN ################################################

k = 10

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, test_size=0.3)
news_knn = KNeighborsClassifier(n_neighbors=k)
news_knn.fit(X_train, y_train)

y_pred = news_knn.predict(X_test)
print('Confusion Matrix')
print(metrics.confusion_matrix(y_true = y_test, y_pred = y_pred))
print('Accuracy = ', metrics.accuracy_score(y_true = y_test, y_pred = y_pred))

# When k = 10, Accuracy =  0.552419990541

#%%

news_knn.get_params()

Ks = np.linspace(1,100,100)
Accuracies = np.zeros(Ks.shape[0])
for i,k in enumerate(Ks):
    news_knn = KNeighborsClassifier(n_neighbors=int(k))
    scores = cross_val_score(estimator = news_knn, X = X, y = y, cv=5, scoring='accuracy')     # cv = 5 ??
    Accuracies[i]  = scores.mean()
    
plt.plot(Ks,Accuracies)
plt.title('Evaluating accuracy of k-values')
plt.xlabel('k value')
plt.ylabel('Accuracy')
plt.show()
    
#%% 
    
plt.plot(Ks,Accuracies)
plt.title('Evaluating accuracy of k-values')
plt.xlabel('k value')
plt.ylabel('Accuracy')
plt.xlim(30,40)
plt.show()

# choose k = 37
# when k = 37, Accuracy =  0.564401702664 - still not that great

# %% Your interpretation

#At the start, I arbitrarily chose k = 10, which yielded an accuracy of 0.55, which is not too impressive.  I evaluated the accuracy at 100 k-values between 1 and 100, which illustrated that the k-value of 37 yielded the maximum accuracy of 0.564, which is only  marginally better.

#%% Task 2.4 SVM ################################################

Xsubset = X[:500][:]
ysubset = y[:500]

X_train, X_test, y_train, y_test = train_test_split(Xsubset, ysubset, random_state=1, test_size=0.3)

news_svm = svm.SVC(kernel='rbf',C=100)
news_svm.fit(X_train, y_train)

y_pred = news_svm.predict(X_test)
print('Confusion Matrix')
print(metrics.confusion_matrix(y_true = y_test, y_pred = y_pred))
print('Accuracy = ', metrics.accuracy_score(y_true = y_test, y_pred = y_pred))

# when C = 100, Accuracy =  0.6

#%%
news_svm.get_params()

Cs = np.linspace(1,500,100)
Accuracies = np.zeros(Cs.shape[0])
for i,C in enumerate(Cs): # what is enumerate?
    print(i,C)
    news_svm = svm.SVC(kernel='rbf', C = C)
    scores = cross_val_score(estimator = news_svm, X = Xsubset, y = ysubset, cv=5, scoring='accuracy')     # cv = 5 ??
    Accuracies[i]  = scores.mean()
    
#%%

plt.plot(Cs,Accuracies)
plt.title('Evaluating accuracy of C')
plt.xlabel('C value')
plt.ylabel('Accuracy')
#plt.xlim(0,10)
plt.show()

# getting a constant accuracy of 0.56 when model above gives accuracy of 0.6
# when i increased the subset, C = 1 was the maximum accuracy, then dropped significantly to a constat value

# why is this?

#%% Task 2.5 Decision Trees ##################################

#labels =["Popular", "Unpopular"]

def splitData(features):
    news_predictors = X # titanic[features].as_matrix()
    news_labels = y # titanic["Survived"].as_matrix()

    # Split into training and test sets
    XTrain, XTest, yTrain, yTest = train_test_split(news_predictors, news_labels, random_state=1, test_size=0.3)
    return XTrain, XTest, yTrain, yTest

#%%

all_features = list(newsdf)[2:60]
XTrain, XTest, yTrain, yTest = splitData(all_features)

min_ss = np.arange(3,50,2)
#min_ss = [3, 4, 5, 6, 10]
max_depths = np.arange(3,50,2)
#max_depths = [3, 5, 10, 15, 20, 25, 30, 40, 50, 60]
#Accuracies = np.zeros(min_ss.shape[0])
#Accuracies = np.zeros(len(min_ss))
#for i,ss in enumerate(min_ss): 
accuracies = []
for ss in min_ss: 
    for depth in max_depths:
        #print(i,ss,depth)
        decisionTree = tree.DecisionTreeClassifier(min_samples_split=ss,max_depth=depth)
        y_pred = cross_val_predict(estimator = decisionTree, X = Xsubset, y = ysubset, cv=5) #change Xsubset to X
        accuracy = metrics.accuracy_score(y_true = ysubset, y_pred = y_pred)
        accuracies.append(accuracy)
        #print('min sample split: '+str(ss)+', max_depth: '+str(depth)+', accuracy: '+str(accuracy))
        
print('max accuracy = '+str(max(accuracies)))

#plt.plot(min_ss,accuracies)
#plt.title('Min Sample Splits')
#plt.show()
#
#plt.plot(max_depths,accuracies)
#plt.title('Max Depths')
#plt.show()


plt.scatter(min_ss, max_depths, s=Accuracies, c=Accuracies) # is this the best way to show this???
##plt.plot(min_ss,Accuracies)
plt.title('Evaluating accuracy of DT models')
#plt.xlabel('Min_samples_split value')
#plt.ylabel('Accuracy')
##plt.xlim(0,10)
plt.show()

#Accuracy on training data=  1.0  # this seems wrong
#Accuracy on test data=  0.566766514268

# when max_depth = 3:
#Accuracy on training data=  1.0
#Accuracy on test data=  0.568469178622

# when max_depth = 10
#Accuracy on training data=  1.0
#Accuracy on test data=  0.569761942299

#min_samples_split=15,max_depth=10
#Accuracy on training data=  1.0
#Accuracy on test data=  0.569730411477