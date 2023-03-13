from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, acf

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer

from models.prophet_model import ProphetModel
from models.ets_model import EtsModel
from models.xgb_model import XgbModel
from models.arima_model import ArimaModel

from scipy import signal

import numpy as np
import pandas as pd
import joblib

def get_features(df):
    # Set date column as the index
    df.set_index('ds', inplace=True)

    # Extract time series features
    features = pd.DataFrame()
    features['mean'] = df.mean()
    features['std'] = df.std()
    features['min'] = df.min()
    features['max'] = df.max()
    features['median'] = df.median()
    features['kurtosis'] = df.kurtosis()
    features['skewness'] = df.skew()
    features['quantile_25'] = df.quantile(q=0.25)
    features['quantile_75'] = df.quantile(q=0.75)
    features['range'] = df.max() - df.min()
    features['interquartile_range'] = features['quantile_75'] - features['quantile_25']
    features['variation_coefficient'] = features['std'] / features['mean']

    # Compute the autocorrelation score
    '''acf_vals  = acf(df['y'], fft=True)
    autocorr_score = max(acf_vals)
    features['autocorr_score'] = autocorr_score
    '''
    lags = range(31)
    autocorr = acf(df, nlags=30)
    autocorr_score = np.abs(np.mean(autocorr))
    features['autocorr_score'] = autocorr_score

    # Identify and count outliers using the interquartile range (IQR) method
    #q1 = df.quantile(0.25)
    q1 = df['y'].quantile(0.25)
    #q3 = df.quantile(0.75)
    q3 = df['y'].quantile(0.75)
    iqr = q3 - q1
    outliers = ((df < (q1 - 1.5 * iqr)) | (df > (q3 + 1.5 * iqr))).sum(axis=1)
    features['outliers'] = outliers

    # Compute the stationarity score
    adf_test = adfuller(df['y'])
    stationarity_score = adf_test[1]
    features['stationarity_score'] = stationarity_score

    # Compute the spectral density score using Welch's method
    freqs, psd = signal.welch(df['y'], nperseg=256)
    spectral_density_score = np.sum(psd)
    features['spectral_density_score'] = spectral_density_score

    # Compute Noise score
    window_size = 30
    detrended = signal.detrend(df['y'], type='linear')
    noise_variance = np.var(detrended)
    residuals = df['y'] - detrended
    noise_score = np.std(residuals)
    features['noise_score'] = noise_score

    # Extract seasonality, trend and residual components using seasonal_decompose
    try:
        result = seasonal_decompose(df, period = 12)
        features['trend'] = result.trend.mean()
        features['residual'] = result.resid.mean()
        features['seasonality'] = result.seasonal.mean()
    except:
        features['seasonality'] = np.nan

    # Print extracted features
    return features

def clean_data(df):
    if 'Unnamed: 0' in df.columns:
            df = df.drop(columns = 'Unnamed: 0', axis = 1)

    df.columns.values[0] = 'ds'
    df.columns.values[1] = 'y'
    
    df = df.reset_index()

    df['ds'] = pd.to_datetime(df['ds'])
    df['ds'] = df['ds'].dt.tz_localize(None)

    #Impute point_value
    df['y'] = df['y'].fillna(df['y'].mean())

    df = df.drop(columns = ['index'])

    return df

def preprocess_data(df):
    # remove features that have all NaN values
    df = df.dropna(axis='columns', how='all')

    return df

def feature_selection(df):
    return df.drop(columns = ['mean', 'std', 'min', 'max', 'median', 'kurtosis', 'skewness', 'quantile_25', 'quantile_75', 'range', 'interquartile_range', 'variation_coefficient'])

def get_label(df):
    # Load the model from a file using joblib
    classf_model = joblib.load('/Users/vishvabalacs/Downloads/ASH/backend/models/ClassificationModel.pkl')

    # Use the loaded model to make predictions
    y_pred = classf_model.predict(df)

    return y_pred[0]

def get_model_by_label(df, date_from, date_to, label):
    if(label == 0):
        return ["PROPHET", ProphetModel(df, date_from, date_to)]
    elif(label == 1):
        return ["ETS", EtsModel(df, date_from, date_to)]
    elif(label == 1):
        return ["XGBOOST", XgbModel(df, date_from, date_to)]
    elif(label == 1):
        return ["ARIMA", ArimaModel(df, date_from, date_to)]