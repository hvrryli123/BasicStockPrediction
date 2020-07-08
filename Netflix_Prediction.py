# Install Dependencies
from alpha_vantage.timeseries import TimeSeries
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split

api_key = 'MGYLDSGKXGE3WRWF'

### Retrieve Stock Data ###
stock = "NFLIX"
ts = TimeSeries(key=api_key, output_format='pandas')
df, meta_data = ts.get_daily_adjusted(symbol = stock, outputsize = "full")
df = df.sort_values(by=["date"])

#Retrieve Adjusted Close Price
df = df[["5. adjusted close"]]
# Varibale to adjust forecast 'n' days into the future
forecast_out = 30
# Create dependent variable shifted 'n' units up
df['Prediction'] = df[['5. adjusted close']].shift(-forecast_out)

### Create the independent data set ###
# Convert the dataframe to numpy array
x = np.array(df.drop(['Prediction'],1))
# Remove the last 'n' rows
x = x[:-forecast_out]

### Create the dependent data set ###
# Convert the dataframe to a numpy array
y = np.array(df['Prediction'])
# Get all of the y value except the last 'n' rows
y = y[:-forecast_out]


# Split data into 80% training and 20% testing
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)

# Create and train SVM model
svr_rbf = SVR(kernel = 'rbf', C=1e3, gamma = 0.1)
svr_rbf.fit(x_train, y_train)

# Testing Model: Score returns the coefficient of determination R^2 of the prediction
svm_confidence = svr_rbf.score(x_test, y_test)
print("svm confidence: ", svm_confidence)

# Create and train the Linear Regression Model
lr = LinearRegression()
# Train the model
lr.fit(x_train, y_train)

# Test LR model
lr_confidence = lr.score(x_test, y_test)
print("lr confidence: ", lr_confidence)

#Set x_forecast equal to the last 30 rows of the original data set from Adj. Close column
x_forecast = np.array(df.drop(['Prediction'], 1))[-forecast_out:]

# Print the LR model predictions for the next 'n' days
lr_prediction = lr.predict(x_forecast)
print(lr_prediction)

# Print the SVR model predictions for the next 'n' days
svm_prediction = svr_rbf.predict(x_forecast)
print(svm_prediction)

