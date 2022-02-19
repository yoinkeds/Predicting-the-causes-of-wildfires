import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn import tree, preprocessing
import sklearn.ensemble as ske
from sklearn.model_selection import train_test_split
import xgboost as xgb


cnx = sqlite3.connect('FPA_FOD_20170508.sqlite')

df = pd.read_sql_query("SELECT FIRE_YEAR,STAT_CAUSE_DESCR,LATITUDE,LONGITUDE,STATE,DISCOVERY_DATE,FIRE_SIZE FROM 'Fires'", cnx)


df['DATE'] = pd.to_datetime(df['DISCOVERY_DATE'] - pd.Timestamp(0).to_julian_date(), unit='D')


df['MONTH'] = pd.DatetimeIndex(df['DATE']).month
df['DAY_OF_WEEK'] = df['DATE'].dt.day_name()
df_orig = df.copy()

le = preprocessing.LabelEncoder()
df['STAT_CAUSE_DESCR'] = le.fit_transform(df['STAT_CAUSE_DESCR'])
df['STATE'] = le.fit_transform(df['STATE'])
df['DAY_OF_WEEK'] = le.fit_transform(df['DAY_OF_WEEK'])

df = df.drop('DATE', axis=1)
df = df.dropna()

X = df.drop(['STAT_CAUSE_DESCR'], axis=1).values
y = df['STAT_CAUSE_DESCR'].values



def set_label(cat):
    cause = 0
    natural = ['Lightning']
    accidental = ['Structure', 'Fireworks', 'Powerline', 'Railroad', 'Smoking', 'Children', 'Campfire', 'Equipment Use',
                  'Debris Burning']
    malicious = ['Arson']
    other = ['Missing/Undefined', 'Miscellaneous']
    if cat in natural:
        cause = 1
    elif cat in accidental:
        cause = 2
    elif cat in malicious:
        cause = 3
    else:
        cause = 4
    return cause


df['LABEL'] = df_orig['STAT_CAUSE_DESCR'].apply(
    lambda x: set_label(x))
df = df.drop('STAT_CAUSE_DESCR', axis=1)

X = df.drop(['LABEL'], axis=1).values
y = df['LABEL'].values
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, random_state=0) 

xg_reg = xgb.XGBClassifier(tree_method='gpu_hist', gpu_id=0, objective ='binary:logistic', colsample_bytree = 0.5, learning_rate = 0.2, max_depth = 20, alpha = 20, n_estimators = 1000, subsample=0.8)
xg_reg = xg_reg.fit(X_train, y_train)
print(xg_reg.score(X_test,y_test))
