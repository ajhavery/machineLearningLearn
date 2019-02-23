import quandl
import pandas as pd
import numpy as np
from sklearn import preprocessing, svm
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LinearRegression
import math

dataframe = quandl.get("WIKI/GOOG")
df = dataframe[['Adj. Open',  'Adj. High',  'Adj. Low',  'Adj. Close', 'Adj. Volume']]
df['PCT_spread'] = (df['Adj. High'] - df['Adj. Low']) / df['Adj. Close'] * 100
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100
df = df[['Adj. Close',  'PCT_spread',  'PCT_change',  'Adj. Volume']]

forecast_col = 'Adj. Close'
df.fillna(value=-99999, inplace=True) #-99999 will be treated as outlier by model
forecast_out = int(math.ceil(0.01 * len(df))) # consider 1% of dataset length as the time for which we can predict
df['label'] = df[forecast_col].shift(-forecast_out)
print(len(df),df.head(15))
df.dropna(inplace=True) #drop any still remaining na

# we will define features (i.e. parameters used for predictions) as X
# and will define predicted values (aka labels) as y

X = np.array(df.drop(['label'],1))
y = np.array(df['label'])

#in machine learning, features are brought in the range (-1,1)
X = preprocessing.scale(X)