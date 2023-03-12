# data-genie-hackathon

## Solution

The first step of the solution is to define a classification model that could choose the best time series model to fit the data. One approach is to fit the time series data into multiple models
and choose the one which has the least Mean Absolute Percentage Error (MAPE). In this solution, the numerical scores of time series characteristics like seasonality, trend, autocorrelation are calculated 
are extracted from the data and a new data set is formed. This dataset is trained in multiple classification models and the best model is used for further functions. The best time series model is 
chosen and the input data is fit, predictions are done using the model.

## Constructing the dataset

A dataset with time series characteristics as features and best model as label is constructed with sample time series data. The time series characteristics considered are

+ Autocorrelation
+ Outliers
+ Stationarity
+ Spectral Density
+ Noise
+ Outliers
+ Trend
+ Residual
+ Seasonality

A numerical score for each of the above characteristics is calculated. Other parameters like mean, standard deviation, min-value, max-value, skewness, kurtosis etc.. are also calculated. These make up the features of our
dataset. 

Now, for the label ie. to choose the best model for each data, multiple time series models are fit into the data and the model with least MAPE value is chosen.
The models considered are

+ Prophet - 0
+ ETS - 1
+ XGBoost - 2
+ ARIMA - 3

The dataset contains 54 entries. It can be extended by taking some more time series data into account.
The dataset - [TSA Dataset](https://drive.google.com/file/d/15lmgaqcEyR4z0js64ThRXEAp_hX5X3Nu/view?usp=share_link) 

## Training the classifier model



