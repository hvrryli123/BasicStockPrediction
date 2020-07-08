# Install Dependencies
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from datetime import datetime
import Facebook_Prediction as FB
import Amazon_Prediction as AMZN
import Apple_Prediction as AAPL
import Google_Prediction as GOOGL
import Netflix_Prediction as NFLX

#### Visualise Data ###
days = pd.date_range(datetime.today(), periods=30).tolist()

fig = plt.figure(figsize = (12, 8))
plt.plot(days, FB.lr_prediction, label = "Facebook", color = "blue", marker = "o")
plt.plot(days, AMZN.lr_prediction, label = "Amazon", color = "orange", marker = "o")
plt.plot(days, AAPL.lr_prediction, label = "Apple", color = "gray", marker = "o")
plt.plot(days, NFLX.lr_prediction, label = "Netflix", color = "red", marker = "o")
plt.plot(days, GOOGL.lr_prediction, label = "Google", color = "green", marker = "o")
sns.set_style("whitegrid")
sns.despine()
plt.xlabel("Date", fontsize=15)
plt.ylabel("Adjusted Closing Price ($USD)", fontsize=15)
plt.title("Linear Regression model of FAANG stock price in the next 30 days", fontsize = 20)
plt.legend()
plt.show()
