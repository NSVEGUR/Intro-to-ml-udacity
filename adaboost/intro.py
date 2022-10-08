#for sample Datasets
from numpy import mean
from sklearn.datasets import make_classification 
#instead of accuracy score cross_val_score cross validates making k folds of the data
#like taking first 10 percent as test and other 90 as training and so on folds
from sklearn.model_selection import cross_val_score
#for specifying k folder fo cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
#adaboost model
from sklearn.ensemble import AdaBoostClassifier

#extract the data into features and labels [X, y]
features, labels = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=6)

model = AdaBoostClassifier()

#rather than fitting since we are using cross folds we use repeated stratified k fold
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

#n_jobs = -1 means using all the processors available i.e., n_jobs implies number of parallel folds to be run at a time to compute
n_scores = cross_val_score(model, features, labels, cv=cv, scoring="accuracy", n_jobs=-1, error_score="raise")


print(f"Accuracy: {round(mean(n_scores))}")