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

#Using the test dataset and with a test_size of 0.3, the model's accuracy is ~98%, meaning that 98% of the images were classified correctly.  As seen in the confusion matrix, the most common mistake made by the model is incorrectly classifying 2 digits as '2' when they were actually a '3', which it did in two cases.
#
#In Task 1.1.5, we evaluate accuracies of the SVM model and found that any C greater than or equal to 6 results in a constant maximum accuracy of 95.4% using cross validation.
#
#In Task 1.1.6, using the raw, unscaled data yielded an accuracy of 0.5, which is significantly lower than the model using the scaled data.  The model misclassified 23 images as showing '5' when they were actually '1'.  Interestingly, this model had the most trouble with handwritten images of 5s.
#
#Evaluating several different values of C for the model on raw data, the plot shows that any C greater than or equal to 6 should yield an accuracy of ~0.475.  This is significantly lower than the accuracy of the model on scaled data.






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


# YOUR INTERPETATION
#Using k-NN with a test_size of 0.3, we see an accuracy of 0.97, which is almost as good as the SVM accuracy of 0.98 in Task 1.1.  The most common mistake, which is seen in the Confusion Matrix, is that the model mis-classified 2 data points as Class 1 when they were actually class 8 and 2 more data points as class 5 when they were actually class 9.
#
#In Task 1.2.5, the variations of k show that there are two k-values that would yield high accuracy values of ~0.945 including k = 3 and k = 6, using cross-validation.  At k-values greater than 10, accuracy decreases steadily. Using k = 6, our accuracy of the k-NN model increased marginally from 0.966667 to 0.974.
#
#In Task 1.6.6, we use the raw data.  Using k = 6, we get an accuracy of 0.988889, which is incredible.  Evidently, the k-NN model does very well on unscaled data.







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

Ks = np.arange(1,80)
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

#%%

k = 37 # based on cross-validation

#X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, test_size=0.3)
news_knn = KNeighborsClassifier(n_neighbors=k)
news_knn.fit(X_train, y_train)

y_pred = news_knn.predict(X_test)
print('Confusion Matrix')
print(metrics.confusion_matrix(y_true = y_test, y_pred = y_pred))
print('Accuracy = ', metrics.accuracy_score(y_true = y_test, y_pred = y_pred))



# %% Your interpretation

#Maximum accuracy achieved: 0.5803
#
#Explanation:
#At the start, I arbitrarily chose k = 10, which yielded an accuracy of 0.565, which is not too impressive.  I evaluated the accuracy at 100 k-values between 1 and 100, which illustrated that the k-value of 37 yielded the maximum cross-validation accuracy of 0.564.  When I ran the k-NN model using k = 37, the accuracy was 0.5803.

#%% Task 2.4 SVM ################################################

Xsubset = X[:5000][:]
ysubset = y[:5000]

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

Cs = np.linspace(1,200,10)
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

#%%
# now run model on entire dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, test_size=0.3)

news_svm = svm.SVC(kernel='rbf',C=100) # change this
news_svm.fit(X_train, y_train)

y_pred = news_svm.predict(X_test)
print('Confusion Matrix')
print(metrics.confusion_matrix(y_true = y_test, y_pred = y_pred))
print('Accuracy = ', metrics.accuracy_score(y_true = y_test, y_pred = y_pred))


#%% Task 2.5 Decision Trees ##################################

decisionTree = tree.DecisionTreeClassifier(max_depth=10, min_samples_split=10)
decisionTree = decisionTree.fit(XTrain, yTrain)

y_pred_train = decisionTree.predict(XTrain)
print('Accuracy on training data= ', metrics.accuracy_score(y_true = yTrain, y_pred = y_pred_train))

y_pred = decisionTree.predict(XTest)
print('Accuracy on test data= ', metrics.accuracy_score(y_true = yTest, y_pred = y_pred))
#renderTree(decisionTree, all_features)




#%%
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

min_ss = np.arange(3,450,20)
#min_ss = [3, 4, 5, 6, 10]
max_depths = np.arange(3,58,5) # maximum depth = number of features/variables
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
        print('min sample split: '+str(ss)+', max_depth: '+str(depth)+', accuracy: '+str(accuracy))
        
print('max accuracy = '+str(max(accuracies)))

#plt.plot(min_ss,accuracies)
#plt.title('Min Sample Splits')
#plt.show()
#
#plt.plot(max_depths,accuracies)
#plt.title('Max Depths')
#plt.show()


#plt.scatter(min_ss, max_depths, s=Accuracies, c=Accuracies) # is this the best way to show this???
###plt.plot(min_ss,Accuracies)
#plt.title('Evaluating accuracy of DT models')
##plt.xlabel('Min_samples_split value')
##plt.ylabel('Accuracy')
###plt.xlim(0,10)
#plt.show()

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

#%% run model again w/ optimized parameters

decisionTree = tree.DecisionTreeClassifier(max_depth=8, min_samples_split=403)
decisionTree = decisionTree.fit(XTrain, yTrain)

y_pred_train = decisionTree.predict(XTrain)
print('Accuracy on training data= ', metrics.accuracy_score(y_true = yTrain, y_pred = y_pred_train))

y_pred = decisionTree.predict(XTest)
print('Accuracy on test data= ', metrics.accuracy_score(y_true = yTest, y_pred = y_pred))
#renderTree(decisionTree, all_features)

# YOUR INTERPRETATION
#1. Which method (k-NN, SVM, Decision Tree) worked best? <br>
#    k-NN max accuracy = 0.5803 <br>
#    SVM max accuracy = 0.546 ### CHECK THIS ### <br>
#    Decision Tree max accuracy = 0.6506 <br>
#+ How did different parameters influence the accuracy? <br>
#    k-NN: k strongly influenced cross-validation accuracy ranging from 0.53 to 0.56 <br>
#    SVM: model accuracy was constant and did not depend on C, which is computationally expensive to determine <br>
#    Decision Tree:  <br>
#+ Which model is easiest to interpret?
#    The k-NN and decision tree models are easiest to intepret primarily because they can be easily visualized.
#+ How would you interpret your results?