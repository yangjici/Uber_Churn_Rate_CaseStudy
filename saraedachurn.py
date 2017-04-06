import numpy as np
import pandas as pd
import sklearn.linear_model as skl
import sklearn.tree as skt
import matplotlib as plt
import seaborn as sns
from pandas.tools.plotting import scatter_matrix

df = pd.read_csv('data/churn_train.csv')
df.info()
df.head()
df.isnull().sum()
df['avg_rating_of_driver'].mean()
df['avg_rating_of_driver'].min()

test = df
test.groupby(by=test['city'])

train = pd.read_csv('data/churn_train.csv',parse_dates = ['last_trip_date','signup_date'],infer_datetime_format=True)
pulled_date = pd.to_datetime('2014-07-01')
train['days_since_trip'] =  (pulled_date - train['last_trip_date']).dt.days
train.head()
y_train_churn = (train['days_since_trip'].values > 30) * 1
y_train_churn

train['driver_experience'] = ['low' if x <4.0 else 'high' for x in train['avg_rating_by_driver']]
