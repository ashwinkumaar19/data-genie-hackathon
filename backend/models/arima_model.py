import statsmodels.api as sm
import numpy as np
import pandas as pd
import pmdarima as pm

class ArimaModel:
    def __init__(self, df, start_date, end_date) -> None:
        self.model = None
        self.df = df
        self.start_date = start_date
        self.end_date = end_date
        self.train_data = self.df[(self.df['ds'] < self.start_date) | (self.df['ds'] > self.end_date)]
        self.test_data = self.df[(self.df['ds'] >= self.start_date) & (self.df['ds'] <= self.end_date)]

    def fit_model(self):
        self.train_data.set_index('ds', inplace=True)

        self.model = pm.auto_arima(self.train_data['y'],    
            seasonal=False,
            suppress_warnings=True,
            error_action='ignore',
            start_p=0, d=None, start_q=0,
            max_p=5, max_d=2, max_q=5,
            test="adf"
        ) 
  
        self.model.fit(self.train_data['y']) 
    
    def make_predictions(self):
        # make predictions on the test data
        predictions = self.model.predict(n_periods=len(self.test_data))
        predictions = predictions.reset_index().rename(columns={'index': 'ds', 0: 'yhat'})

        return predictions

    def get_mape(self):
        y_true, y_pred = np.array(self.test_data['y']), self.model.predict(n_periods=len(self.test_data))
        
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    def get_forecast(self, n):
        self.df = self.df.reset_index()
        self.df.set_index('ds', inplace = True)

        self.model = pm.auto_arima(self.df['y'],    
            seasonal=False,
            suppress_warnings=True,
            error_action='ignore',
            start_p=0, d=None, start_q=0,
            max_p=5, max_d=2, max_q=5,
            test="adf"
        ) 
  
        self.model.fit(self.df['y']) 

        predictions = self.model.predict(n_periods=n)
        predictions = predictions.reset_index().rename(columns={'index': 'ds', 0: 'yhat'})

        return predictions