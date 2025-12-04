**Stock Price Prediction — Linear Regression vs SVR (Using yfinance)**

This project builds a simple machine-learning model to predict future stock closing prices using two algorithms:
Linear Regression
Support Vector Regressor (SVR) with RBF kernel
The program downloads Apple (AAPL) historical stock data, prepares a shifted prediction column, trains the models, compares their performance, and outputs a 30-day price forecast.

Features:
Fetches historical stock data using yfinance

Uses only the Close price for prediction

Creates a 30-day-ahead forecast target (Prediction column)

Compares SVR vs Linear Regression

Prints R² scores and future predictions

Simple, clean, and easy to modify

📦 Dependencies

Install the following Python packages:

numpy
pandas
yfinance
scikit-learn

▶️ Running the Script
Run:
python3 main.py

The script downloads data from:
AAPL from 2010-01-01 to 2025-12-02

You will see:
Total number of rows downloaded
First 5 rows
Data after selecting only Close column
Last 5 rows showing the shifted Prediction column

Model confidence scores
30-day forecast from SVR and Linear Regression

** Example Output (Your Actual Output)**
**Dataset size**
4003

**Sample raw data**
            Close      High       Low      Open     Volume
2010-01-04  6.4183    6.4330    6.3694    6.4009   493729600
2010-01-05  6.4294    6.4657    6.3955    6.4360   601904800
...

**Close column**
2010-01-04  6.418383
2010-01-05  6.429481
2010-01-06  6.327210
2010-01-07  6.315513
2010-01-08  6.357502

**Last rows (Prediction = future shifted values)**
Date         Close        Prediction
2025-11-24   275.92       NaN
2025-11-25   276.97       NaN
2025-11-26   277.55       NaN
2025-11-28   278.85       NaN
2025-12-01   283.10       NaN

** Model Performance (Actual Output)**
**SVM Confidence (R² Score)**
0.9820761610203554

**Linear Regression Confidence**
0.98402024910258

Both models perform very well due to the smooth nature of the Close price time-series.

30-Day Forecast (Actual Predictions)
SVR Predictions
[139.94, 136.45, 234.31, 199.55, 136.22, 134.19, ... , 134.20]


(30 values — small excerpt shown above)
Linear Regression Predictions
[266.34, 266.88, 262.51, 263.65, 266.93, 272.99, ... , 287.73]


(30 values — small excerpt shown above)

📘 How the Model Works
1. Prepare dataset
df = df[['Close']]
df['Prediction'] = df['Close'].shift(-30)

2. Create features and labels
X = df.drop(['Prediction'], axis=1).values[:-30]
y = df['Prediction'].values[:-30]

3. Split into training & testing
train_test_split(X, y, test_size=0.2)

4. Train SVR
SVR(kernel='rbf', C=1e3, gamma=0.1)

5. Train Linear Regression
LinearRegression()

6. Forecast next 30 days
x_forecast = df.drop(['Prediction'], axis=1).values[-30:]
svr_rbf.predict(x_forecast)
lr.predict(x_forecast)


This does not affect your model.

The model works but can be improved using scaling, more features, or time-series methods.
