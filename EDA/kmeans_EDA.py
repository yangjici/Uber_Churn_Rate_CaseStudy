#GOAL: reduce churn rate and increase ridership
#prioritize explanatory factors rather than predictive
#dont engineer features that looking into the future
#training model on the target and predict target: leakage
#lesson: get a model working first
#dont spend too much time cleaning data

import numpy as np
import pandas as pd
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.cluster import KMeans

''' parsing dates into intergers'''

train = pd.read_csv('data/churn_train.csv',parse_dates = ['last_trip_date','signup_date'],infer_datetime_format=True)
pulled_date = pd.to_datetime('2014-07-01')

train['days_since_trip'] =  (pulled_date - train['last_trip_date']).dt.days
train.head()
y_train_churn = (train['days_since_trip'].values > 30) * 1
y_train_churn

train['churn'] = y_train_churn

'''62.42 percentage of user churned'''



'replacing nan in phone with other'

train['phone'] = train['phone'].fillna('Other')

''' dummify cities'''

def make_dummies(data, columns):
   for column in columns:
       dummies = pd.get_dummies(data[column])
       data = pd.concat([data, dummies], axis=1)
       data = data.drop(column, axis=1)
   return data

train=make_dummies(train,columns=['city','phone'])


def is_weekend(data, columns):
   for col in columns:
       #weekend if day >= 5 (sat, sun)
       data[col+'_is_weekend']=data[col].dt.dayofweek >= 5
   return data

#add columns identifying whether signup date and last trip date were on weekends

train = is_weekend(train, columns = ['last_trip_date','signup_date'])



#drop several dummy variables

train = train.drop(['Astapor','Other','signup_date','signup_date_is_weekend','last_trip_date','last_trip_date_is_weekend'],axis = 1)


#convert boolean to number

train['luxury_car_user'] = [1 if x else 0 for x in train['luxury_car_user']]






'''several thoughts
1. bad rating of driver signal bad experience and high churn rate?
2. too many surge pricing trips and high surge price leading to churn?
3. frequent ridership less likely to churn?
4. weekday rider who churn are commuters who prefer cheaper way to commute?
5. Average distance is a predictor of the need for trainsit, higher distance
less likely to churn?
6. average rating by driver is indicator of the person's personality
lower rating suggest more impulsive and unpleasant (frequent ditching/no show)
7. Interaction between by driver and rating of driver might be an interesting
source of insight
'''

'''
investigate correlation between rating by driver and of driver
'''

scatter_matrix(train[['avg_rating_by_driver' , 'avg_rating_of_driver']],alpha = 0.2, figsize=(2,2),diagonal= 'kde')

'''
user's rating by driver and of driver is highly correlated
'''


'''
Imput missing rider experience to be the same as driver experience
given the two rating are highly correlated
'''

train['avg_rating_of_driver']=[ by_driver if np.isnan(of_driver) else of_driver  for of_driver, by_driver in  zip(train['avg_rating_of_driver'],train['avg_rating_by_driver'])]


'''
investigate churn rate and it's relation to rider with lower rating by the driver
'''

'''bin the users into three groups based on the rating by driver driver:
low, and high'''

train['bad_driver_experience'] = [1 if x <4.0 else 0 for x in train['avg_rating_by_driver']]

train['churn'].groupby(train['bad_driver_experience']).mean()

'''
user_experience
high    0.618981
low     0.786517
'''

'''
high proportion of churn for user with lower rating from driver
'''

'''
investigate churn rate and it's relation to rider with lower rating of the driver
'''


train['bad_user_experience'] = [1 if x <4.0 else 0 for x in train['avg_rating_of_driver']]

train['churn'].groupby(train['bad_user_experience']).mean()



'''
impute the rest of the rating by their Average

'''

def impute_by_av(data,columns):
    for col in columns:
        data[col] = [np.mean(data[col]) if np.isnan(x) else x for x in data[col]]
    return data

train = impute_by_av (train,['avg_rating_by_driver','avg_rating_of_driver'])



'''
K-means clustering to identify certain rider groups

Scaling data to weight each feature similarly

Features we used to cluster our users with:

'avg_dist', 'avg_rating_by_driver', 'avg_rating_of_driver',
'avg_surge', 'surge_pct', 'trips_in_first_30_days',
'luxury_car_user', 'weekday_pct'

'''

y= train['churn']

X = train.drop('churn',axis=1)

X_just= X.drop(['iPhone','bad_driver_experience','bad_user_experience',"King's Landing",'Winterfell','Android','days_since_trip',"kmeans"],axis=1)

X_scaled = preprocessing.scale(X_just)

'''cluster with three groups '''

cluster = KMeans(n_clusters=3)

cluster.fit(X_scaled)

train["kmeans"] = cluster.labels_

train.groupby(train['kmeans']).mean()

'''
out of the group with highest churn rate, they have the lowest average distance,
,  highest average percentage of trips taken with surge and highest price surge percentage on average for each trip

'''


''' cluster with two groups '''


cluster = KMeans(n_clusters=2)

cluster.fit(X_scaled)

train["kmeans"] = cluster.labels_

res = train.groupby(train['kmeans'])[['avg_dist','avg_surge','surge_pct','trips_in_first_30_days','luxury_car_user','weekday_pct','churn']].mean()



'''
we replicated the results from the last cluster

        avg_dist  avg_surge  surge_pct  trips_in_first_30_days  \
kmeans
0       5.873880   1.031743    4.44767                2.357655
1       4.629292   1.683022   70.90844                1.197815

        luxury_car_user  weekday_pct     churn
kmeans
0              0.392652    62.284314  0.614363
1              0.143934    41.034401  0.762622


From the clustered groups, we found that weekend users are the ones experiencing majority of the surge and pays higher surge percentage per trip, we are now motivated to investigate whether weekend has disportional amount of surge that is causing the users who use the service to churn



does majority of surge occurs in the weekend?

classifiy those who has less than 40 percentage of weekday usage to be weekend-mainly users'''


X["weekend_user"] = [1 if x<40 else 0 for x in X["weekday_pct"] ]


X.groupby('weekend_user')[['avg_dist','surge_pct','avg_surge','trips_in_first_30_days']].mean()

'''
              avg_dist  surge_pct  avg_surge  trips_in_first_30_days
weekend_user
0             5.831786   7.549646   1.062972                2.612876
1             5.653029  13.323777   1.115887                1.146153
'''

''' yes we indeed found disportionally higher percentage of surge percent


App platform user differences

User with android app has significantly higher churn rate than non-android apps '''

train.groupby('Android')['churn'].mean()

'''
Android
0    0.552224
1    0.791089
Name: churn
'''

'''Problems with Android app? '''



''' plotting and visualizating'''




fig, ax_list = plt.subplots(3, 2)

for col,subp in zip(res.columns.values[1:],ax_list.flatten()):
    subp.bar(res['kmeans'], res[col])
    subp.set_title(col)
