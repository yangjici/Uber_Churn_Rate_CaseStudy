import numpy as np
import pandas as pd
from datetime import datetime
import sklearn.linear_model as skl
import sklearn.tree as skt
import matplotlib as plt
from pandas.tools.plotting import scatter_matrix
import seaborn as sns
from sklearn.ensemble.RandomForest import

datetime_object = datetime.strptime()
df = pd.read_csv('data/churn_train.csv')

train = pd.read_csv('data/churn_train.csv',parse_dates = ['last_trip_date','signup_date'],infer_datetime_format=True)
pulled_date = pd.to_datetime('2014-07-01')
train['days_since_trip'] =  (pulled_date - train['last_trip_date']).dt.days
train.head()
y_train_churn = (train['days_since_trip'].values > 30) * 1
y_train_churn

tree1 = skt.DecisionTreeClassifier()
log1 = skl.LogisticRegression()
lin1 = skl.LinearRegression()
