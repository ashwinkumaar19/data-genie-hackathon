from tracemalloc import start
import numpy as np
import pandas as pd
from statsmodels import test

from xgboost import XGBRegressor
from datetime import timedelta

class XgbModel:
    def __init__(self, df, start_date, end_date) -> None:
        self.model = None
        self.df = df
        self.start_date = start_date
        self.end_date = end_date
        self.train_data = self.df[(self.df['ds'] < self.start_date) | (self.df['ds'] > self.end_date)]
        self.test_data = self.df[(self.df['ds'] >= self.start_date) & (self.df['ds'] <= self.end_date)]

    def create_features(self, df, target_variable):
        df['date'] = df.index
        df['hour'] = df['date'].dt.hour
        df['dayofweek'] = df['date'].dt.dayofweek
        df['quarter'] = df['date'].dt.quarter
        df['month'] = df['date'].dt.month
        df['year'] = df['date'].dt.year
        df['dayofyear'] = df['date'].dt.dayofyear
        df['dayofmonth'] = df['date'].dt.day
        df['weekofyear'] = df['date'].dt.weekofyear
        
        X = df[['hour','dayofweek','quarter','month','year',
            'dayofyear','dayofmonth','weekofyear']]
        
        if target_variable:
            y = df[target_variable]
            return X, y

        return X

    def fit_model(self):
        # Make sure that you have the correct order of the times 
        self.df = self.df.sort_values(by='ds', ascending=True)

        # Set Datetime as index
        self.train_data = self.train_data.set_index('ds')
        self.test_data = self.test_data.set_index('ds')

        trainX, trainY = self.create_features(self.train_data, target_variable='y')
        testX, testY = self.create_features(self.test_data, target_variable='y')

        self.model = XGBRegressor(objective= 'reg:linear', n_estimators=1000)

        self.model.fit(trainX, trainY,
                 eval_set=[(trainX, trainY), (testX, testY)],
                 early_stopping_rounds=50,
                 verbose=False)

    def make_predictions(self):
        testX, testY = self.create_features(self.test_data, target_variable='y')
        
        predictions = self.model.predict(testX)
        predictions = pd.Series(predictions)

        # Generate a datetime series from the start date to the end date
        date_series = pd.Series(pd.date_range(start=self.start_date, end=self.end_date))

        # Concatenate the series along axis 1 and name the resulting columns
        predictions = pd.concat([date_series, predictions], axis=1, keys=['ds', 'yhat'])
        
        return predictions

    def get_mape(self):
        testX, testY = self.create_features(self.test_data, target_variable='y')
        
        y_true, y_pred = np.array(testY), self.model.predict(testX)
        
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    def get_forecast(self, n):
        # Generate date range
        date_range = pd.date_range(start = self.test_data.index[-1] + timedelta(days=1) , periods=n)

        # Create dataframe
        test_df = pd.DataFrame({'ds': date_range, 'y': [1]*n})
        test_df = test_df.set_index('ds')
        
        testX, testY = self.create_features(test_df, target_variable='y')

        predictions = self.model.predict(testX)

        # Concatenate the series along axis 1 and name the resulting columns
        predictions = pd.concat([pd.Series(date_range), pd.Series(predictions)], axis=1, keys=['ds', 'yhat'])

        return predictions