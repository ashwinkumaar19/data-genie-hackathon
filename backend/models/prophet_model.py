from prophet import Prophet
import numpy as np

class ProphetModel:
    def __init__(self, df, start_date, end_date) -> None:
        self.model = None
        self.df = df
        self.start_date = start_date
        self.end_date = end_date
        self.df['ds'].dt.tz_localize(None)
        self.train_data = self.df[(self.df['ds'] < self.start_date) | (self.df['ds'] > self.end_date)]
        self.test_data = self.df[(self.df['ds'] >= self.start_date) & (self.df['ds'] <= self.end_date)]

    def fit_model(self):
        self.model = Prophet()
        self.model.fit(self.train_data)

    def make_predictions(self):
        # Use the model to make predictions on the test data
        forecast = self.model.predict(self.test_data)

        # Display the forecasted values
        return (forecast[['ds', 'yhat']])
    
    def get_mape(self):
        y_true, y_pred = np.array(self.test_data['y']), self.model.predict(self.test_data)['yhat']
        
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    def get_forecast(self, n):
        self.model = Prophet()
        self.model.fit(self.df)

        # Generate a dataframe with the dates of the forecast
        future = self.model.make_future_dataframe(periods=n)

        # Generate a forecast for the future dates
        forecast = self.model.predict(future)

        # Print the forecast
        return forecast.iloc[-n:][['ds', 'yhat']]

