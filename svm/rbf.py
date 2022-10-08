#!/usr/bin/python3

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import numpy as np


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()


#########################################################
### your code goes here ###

clf = SVC(kernel="rbf", C=10000.)


#########################################################

#########################################################
'''
You'll be Provided similar code in the Quiz
But the Code provided in Quiz has an Indexing issue
The Code Below solves that issue, So use this one
'''

#shortening the train dataset for faster computation
# features_train = features_train[:int(len(features_train)/100)]
# labels_train = labels_train[:int(len(labels_train)/100)]

#train
clf.fit(features_train, labels_train)

#test
# labels_pred = clf.predict(features_test)

# #predicting for certain dataset xth element 0->Sara, 1->Chris
# labels_pred = clf.predict(features_test[50].reshape(1, -1))
# print(labels_pred)

# #predicting Chris and Sara predicted emails 
labels_pred = clf.predict(features_test)
print(f"Chris emails: {np.count_nonzero(labels_pred == 1)}")
print(f"Sara emails: {np.count_nonzero(labels_pred == 0)}")


# accuracy = accuracy_score(labels_pred, labels_test)
# print(f'Accuracy: {accuracy}')