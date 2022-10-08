import sys
sys.path.append("../tools/")
from email_preprocess import preprocess
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score

### features_train and features_train are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

model = AdaBoostClassifier(n_estimators=300, learning_rate=1.3)
model.fit(features_train, labels_train)

labels_pred = model.predict(features_test)

accuracy = accuracy_score(labels_pred, labels_test)

print(f"Accuracy: {accuracy}")