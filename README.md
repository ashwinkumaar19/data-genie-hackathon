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

Hyperparameter tuning was done using GridSearchCV, class imbalance was handled using RandomOverSampler and the data was fit to the models. Random Forest Classifier came out with the best accuracy score. Hence this model will be used further to choose the best time series model.

![image](https://user-images.githubusercontent.com/77486930/224581394-7f4b7a8b-4fe5-4a85-b0b0-6925c4627fac.png)

The code for dataset creation and training the classifier is given in the data-genie.ipynb file in this repository.

## Handling the input time series data

Time series characteristics are extracted from the input time series data. It is then provided to the RandomForestClassifier and the best model is predicted. The input data is then split into test and train data. The training data is trained using the time series model that was predicted.

## REST API

A simple UI is written that inputs any time series data. The API processes this data and forecasts point values, and the output is plotted. The underlying REST API uses the pre-trained classifier model to find the best time series model. FastAPI was used as the backend framework for the API. Frontend was done using ReactJS.
The files in the code are explained below

Inputs for the API include 

+ date_from
+ date_to 
Predictions are done from the start date to end date.

+ period
Number of points to forecast into the future

<img width="1440" alt="Screenshot 2023-03-13 at 8 41 30 AM" src="https://user-images.githubusercontent.com/77486930/224599687-ac4058fb-1f38-48b5-94de-968eff32cfca.png">



### Models
This folder contains a class for each time series model that was taken into account. The functions in these files are
+ fit_model - fit the data into the model
+ get_predictions - get predictions from start_date to end_date
+ get_mape - get MAPE value
+ get_forecast - get forecast data based on the "period" input

### Utility
This file contains functions which helps in
+ get_features - get the time series characteristics from the input data
+ clean_data 
+ pre-process the data
+ feature selection
+ get_label - use the pre-trained classifier to choose the best time series model for the data

### App
This file is responsible for handling the incoming request, parse the data, use the utility functions to predict/forecast. A sample output 
<img width="1440" alt="image" src="https://user-images.githubusercontent.com/77486930/224600988-c2caa5a0-36b4-4e17-8b88-206bfdbe1689.png">


This output is for a sample time series data. Predictions are done from 2016-12-01 to 2016-12-27. Both actual and predicted point_values are plotted in the graph. The time series model and the associated MAPE value is shown.

API is also written to forecast based on the "period" input given by the user.


## Notes

+ The code for creating the dataset, training the classifier model was written in cloud. (Google Colab)
+ The model is located in the "models" folder in this repo.

## How to run

### Backend

+ Install all the required modules from the requirements.txt
+ Run using
> uvicorn app:app --reload

### Frontend

+ Install required node_modules
+ Run using
> npm start

