import statsmodels.api as sm
import numpy as np
import pandas as pd

class EtsModel:
    def __init__(self, df, start_date, end_date) -> None:
        self.model = None
        self.df = df
        self.start_date = start_date
        self.end_date = end_date
        self.train_data = self.df[(self.df['ds'] < self.start_date) | (self.df['ds'] > self.end_date)]
        self.test_data = self.df[(self.df['ds'] >= self.start_date) & (self.df['ds'] <= self.end_date)]

    def fit_model(self):
        print(self.train_data)
        self.train_data.set_index('ds', inplace=True)

        self.model = sm.tsa.ExponentialSmoothing(self.train_data['y'], trend='add', seasonal='add', seasonal_periods=12).fit()
    def make_predictions(self):
        # make predictions on the test data
        predictions = self.model.forecast(len(self.test_data))
        predictions = predictions.reset_index().rename(columns={'index': 'ds', 0: 'yhat'})

        return predictions

    def get_mape(self):
        y_true, y_pred = np.array(self.test_data['y']), self.model.forecast(len(self.test_data))
        
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    def get_forecast(self, n):
        self.df = self.df.reset_index()
        self.df.set_index('ds', inplace = True)

        self.model = sm.tsa.ExponentialSmoothing(self.df['y'], trend='add', seasonal='add', seasonal_periods=12).fit()

        predictions = self.model.forecast(n)
        predictions = predictions.reset_index().rename(columns={'index': 'ds', 0: 'yhat'})

        return predictions