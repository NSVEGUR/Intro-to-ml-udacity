#!/usr/bin/python3
import os
import joblib
import sys
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath("../tools/"))
from feature_format import featureFormat, targetFeatureSplit


### read in data dictionary, convert to numpy array
data_dict = joblib.load( open("../final_project/final_project_dataset.pkl", "rb") )
features = ["salary", "bonus"]
#removing total row in the data since it is just an outlier given by spread sheet and we considered it as a data point
data_dict.pop("TOTAL", 0)
data = featureFormat(data_dict, features)


### your code below
for point in data:
    salary = point[0]
    bonus = point[1]
    plt.scatter( salary, bonus )

plt.xlabel("salary")
plt.ylabel("bonus")
plt.show()

