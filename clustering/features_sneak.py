import joblib
import pandas as pd

### load in the dict of dicts containing all the data on each person in the dataset
data_dict = joblib.load( open("../final_project/final_project_dataset.pkl", "rb") )
### there's an outlier--remove it! 
data_dict.pop("TOTAL", 0)


df = pd.DataFrame(data_dict)
df.loc["salary", :] = pd.to_numeric(df.loc["salary", :], errors='coerce')
print(f"Minimum: {df.loc['salary', :].min(skipna=True)}")
print(f"Maximum: {df.loc['salary', :].max(skipna=True)}")
