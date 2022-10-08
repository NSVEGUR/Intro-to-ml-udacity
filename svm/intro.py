from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import numpy as np

features_train = np.array([[-1, -2], [1, 3], [-1, 2], [2, -1], [10, 9], [-10, -6], [6, -3], [-30, 4], [40, 10], [4, 13]])
labels_train = np.array([-1, 1, -1, -1, 1, -1, -1, -1, 1, 1])

features_test = np.array([[-5, 1], [5, 7], [13, -11], [-10, -100]])
labels_test = [-1, 1, -1, -1]

clf = SVC()
clf.fit(features_train, labels_train)
labels_pred = clf.predict(features_test)
print(f'Predicted: {labels_pred}')
accuracy = accuracy_score(labels_pred, labels_test)
print(f'Accuracy: {accuracy}')

