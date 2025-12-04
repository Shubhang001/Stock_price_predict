# Program to predict stock prices using ML Algorithms--> Linear Regression VS Support Vector Regressor

#Install the dependencies
import yfinance as yf
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split

#Get the stock data
df = yf.download("AAPL", start="2010-01-01", end="2025-12-02")
print(df.shape[0])
print(df.head(5))

df = df[['Close']]
print(df.head(5))

#Variable for predicting 'n' days in future
forecast_out = 30
df['Prediction']= df[['Close']].shift(-forecast_out)
print(df.tail(5))

#Create a independent dataset (X)
#Convert the dataframe to a numpy array for Close Data
X = np.array(df.drop(['Prediction'], axis=1))   #would return a list of lists

#Remove the last 'n' Rows
X= X[:-forecast_out]
print(X)

#Create a dependent dataset (y) 
#Convert dataframe to a numpy array(All values including the NaN's)
y= np.array(df['Prediction'])  #Return just the list

#Remove the last 'n' Rows
y= y[:-forecast_out]
print(y)


#Splitting the data into 80% training and 20% testing
x_train, x_test, y_train, y_test= train_test_split(X,y,test_size= 0.2)

#Create and train Support Vector Machine(Regressor) - SVM
svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
svr_rbf.fit(x_train, y_train)

#Testing model: Score returns the coefficient of determination R^2 of the prediction, best possible score is 1.0
svm_confidence = svr_rbf.score(x_test, y_test)
print("SVM Confidence: ", svm_confidence)

#Create and train the Linear Regression Model
lr = LinearRegression()
#Train the model
lr.fit(x_train, y_train)

#Testing Model for linear regression algorithm
lr_confidence =  lr.score(x_test,y_test)
print("lr Confidence: ", lr_confidence)

#Set x_forecast equal to last 30 rows of the original dataset from Close Column
x_forecast = np.array(df.drop(['Prediction'],axis= 1))[-forecast_out:]
print(x_forecast[:5])

#Print Support Vector Regressor model predictions for the next 'n' days
svm_prediction = svr_rbf.predict(x_forecast)
print(svm_prediction)

#Print Support Vector Regressor model predictions for the next 'n' days
lr_prediction = lr.predict(x_forecast)
print(lr_prediction)