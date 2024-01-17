import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import plotly.offline as py
import seaborn as sns

import matplotlib.ticker as mtick
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.tree import ExtraTreeRegressor
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

df=pd.read_csv('deploy_df.csv')

df.drop('Unnamed: 0',axis=1,inplace=True)

x=df.drop('Price',axis=1)
y=df['Price']

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=1)

import xgboost as xgb
xgb_model=xgb.XGBRegressor()
xgb_model.fit(X_train,y_train)

y_predict=xgb_model.predict(X_test)

import pickle
pickle.dump(xgb_model, open('model.pkl','wb'))
model=pickle.load(open('model.pkl','rb'))
