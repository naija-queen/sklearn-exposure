import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score


dataset_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
data = pd.read_csv(dataset_url, sep = ';')

''' Describes the data '''
print(data.head())
print(data.shape)
print(data.describe())

print()

''' Splitting data into training and test sets '''
y = data.quality
X = data.drop('quality', axis = 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,
                                                    random_state = 123,
                                                    stratify = y)

''' Data Preprocessing '''
''' fitting the Transformer API '''
scaler = preprocessing.StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
print(X_train_scaled.mean(axis = 0))
print(X_train_scaled.std(axis = 0))

print()

''' applying transformer to test data '''
X_test_scaled = scaler.transform(X_test)
print(X_test_scaled.mean(axis = 0))
print(X_test_scaled.std(axis = 0))

print()

''' create cross-validation pipeline with preprocessing and model '''
pipeline = make_pipeline(preprocessing.StandardScaler(), 
                         RandomForestRegressor(n_estimators=100))
