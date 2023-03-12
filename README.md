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

The dataset constructed is fed to some classifier models and the best onw with highest accuracy is chosen. Classifier models the dataset was input into are

+ Random Forest
+ AdaBoost
+ Gradient Boosting
+ Logistic Regression
+ Gaussian Naive Bayes
+ Decision Tree
+ K-Nearest Neighbors
+ Multi-layer Perceptron
+ Support Vector Machine

Hyperparameter tuning was also done and the data was fit to the models. Random Forest Classifier came out with the best accuracy score. Hence this model will be used further to choose the best time series model.

## Handling the input time series data

Time series characteristics are extracted from the input time series data. It is then provided to the RandomForestClassifier and the best model is predicted. The input data is then split into test and train data. The training data is trained using the time series model that was predicted.

## REST API

A simple UI is written that inputs any time series data. The API processes this data and forecasts point values, and the output is plotted. The underlying REST API uses the pre-trained classifier model to find the best time series model. 

## Notes

+ The code for creating the dataset, training the classifier model was written in cloud. (Google Colab)
+ The model was saved using joblib in the local storage.
+ The saved model is used in writing the API.


