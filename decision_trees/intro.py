from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import numpy as np

#creating features and labels to test
features_train = np.array([[-1, -2], [1, 3], [-1, 2], [2, -1], [10, 9], [-10, -6], [6, -3], [-30, 4], [40, 10], [4, 13]])
labels_train = np.array([-1, 1, -1, -1, 1, -1, -1, -1, 1, 1])

#using decision tree classifier
clf = DecisionTreeClassifier()

#train the classifier with training data by fitting
clf.fit(features_train, labels_train)

#prediction
features_test = np.array([[-5, 1], [5, 7], [13, -11], [-10, -100]])
labels_pred = clf.predict(features_test)
print(f'Predicted: {labels_pred}')

#calculating accuracy score from the metrics of sklearn
labels_test = [-1, 1, -1, -1] #original labels of the test data
accuracy = accuracy_score(labels_test, labels_pred)
print(f'Accuracy: {accuracy}')