from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd


rf = RandomForestClassifier(n_estimators=100,oob_score=True)
train['avg_rating_of_driver']=[ by_driver if np.isnan(of_driver) else of_driver for of_driver, by_driver in  zip(train['avg_rating_of_driver'],train['avg_rating_by_driver'])]

pipeline = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', SGDClassifier()),
])
predicted = pipeline.fit(Xtrain).predict(Xtrain)
# Now evaluate all steps on test set
predicted = pipeline.predict(Xtest)
pipe = Pipeline(steps=[('forest',rf)])
    prediction = rf.predict(X)
    rf.score(X, y)
pipeline.fit()

#df_output = pd.DataFrame({'SalePrice': y_prediction}, index=df_test['SalesID'])
#df_output.to_csv('answers.csv')
