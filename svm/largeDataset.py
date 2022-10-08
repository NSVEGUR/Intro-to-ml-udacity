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


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()


#########################################################
### your code goes here ###


#train
clf = SVC(kernel='linear')
t0 = time()
clf.fit(features_train, labels_train)
print(f'Time to train: {time() - t0} s')

#test
t0 = time()
labels_pred = clf.predict(features_test)
print(f'Time to train: {time() - t0} s')
accuracy = accuracy_score(labels_pred, labels_test)
print(f'Accuracy: {accuracy}')