from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd


train['avg_rating_of_driver']=[ by_driver if np.isnan(of_driver) else of_driver  for of_driver, by_driver in  zip(train['avg_rating_of_driver'],train['avg_rating_by_driver'])]
