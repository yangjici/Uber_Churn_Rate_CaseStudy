import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
import sklearn.tree as skt
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


#import data with dates modified
train = pd.read_csv('data/churn_train.csv',parse_dates = ['last_trip_date','signup_date'],infer_datetime_format=True)
pulled_date = pd.to_datetime('2014-07-01')
train['days_since_trip'] = (pulled_date - train['last_trip_date']).dt.days

#impute driver ratings
# train['avg_rating_of_driver']=['by_driver' if np.isnan(of_driver) else of_driver  for of_driver, by_driver in  zip(train['avg_rating_of_driver'],train['avg_rating_by_driver'])]self.fail('message')

def is_weekend(data, columns):
   for col in columns:
       #weekend if day >= 5 (sat, sun)
       data[col+'_is_weekend']=data[col].dt.dayofweek >= 5
   return data
train = is_weekend(train, columns = ['last_trip_date','signup_date'])
train['phone'] = train['phone'].fillna('Other')
train['luxury_car_user'] = train['luxury_car_user']*1
train['last_trip_date_is_weekend'] = train['last_trip_date_is_weekend']*1
train['signup_date_is_weekend'] = train['signup_date_is_weekend']*1
#making dummies
def make_dummies(data, columns):
   for column in columns:
       dummies = pd.get_dummies(data[column])
       data = pd.concat([data, dummies], axis=1)
       data = data.drop(column, axis=1)
   return data
train = train.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)
new = make_dummies(train,columns=['city','phone'])
#new = fix trips in first 30 days
y_train_churn = (new['days_since_trip'].values > 30) * 1
y_train_churn = np.array(y_train_churn)
X = new
X = X.drop(['days_since_trip'],axis=1)
X = X.drop(['last_trip_date'],axis=1)
X = X.drop(['signup_date'],axis=1)
X = X.drop(['trips_in_first_30_days'],axis=1)

#,index=y_train_churn[:,0]
#create train/test data
X_train, X_test, y_train, y_test = train_test_split(X,y_train_churn)

log1 = LogisticRegression()
lin1 = LinearRegression()

#Random Forest
rf1 = RandomForestClassifier(n_estimators=100,oob_score=True)
rf1.fit(X_train,y_train)

for f in range(X.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
