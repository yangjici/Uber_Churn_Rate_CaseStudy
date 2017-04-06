from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd


train['avg_rating_of_driver']=[ by_driver if np.isnan(of_driver) else of_driver  for of_driver, by_driver in  zip(train['avg_rating_of_driver'],train['avg_rating_by_driver'])]



#df_output = pd.DataFrame({'SalePrice': y_prediction}, index=df_test['SalesID'])
#df_output.to_csv('answers.csv')
