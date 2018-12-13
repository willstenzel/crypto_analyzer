import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn import neighbors
from sklearn.metrics import mean_squared_error
from math import sqrt
import matplotlib.pyplot as plt



tweet_df = pd.read_pickle("/Users/stenzel/Documents/EECE2300/crypto_analyzer/src/data/intermediate/proccesed_normalized_BitcoinTweets.pkl")
price_df = pd.read_pickle("/Users/stenzel/Documents/EECE2300/crypto_analyzer/src/data/intermediate/formatted_price_data.pkl")

# Get the time of the data set that begins latest
startTime = max(tweet_df.index[0], price_df.index[0])

# keep only rows that have the share the same date times
price_df = price_df[startTime:tweet_df.index[-1]]
tweet_df = tweet_df[startTime:]

# create a composite data frame and canadians both the price and sentient data
composite_df = pd.concat([tweet_df, price_df], axis=1)
print(composite_df)
# split the data into a training and testing
train, test = train_test_split(composite_df, test_size=0.3)

x_train = train.drop('Price', axis=1)
y_train = train['Price']

x_test = train.drop('Price', axis=1)
y_test = train['Price']

# scale the features to normalize the data
scalar = MinMaxScaler(feature_range=(0, 1))

x_train_scaled = scalar.fit_transform(x_train)
x_train = pd.DataFrame(x_train_scaled)

x_test_scaled = scalar.fit_transform(x_test)
x_test = pd.DataFrame(x_test_scaled)



mse_val = [] # stores the mean squared error values for graphing

# loop through different values of K to find the lowest error
for k in range(1, 51):
    model = neighbors.KNeighborsRegressor(n_neighbors=k)
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    error = sqrt(mean_squared_error(y_test, pred))
    mse_val.append(error)
    #print('RMSE value for k= ', k, 'is:', error)

# initialize a new figure
fig, ax = plt.subplots()

plt.title('K-Nearest Neighbors: MSE vs K value', y=1.00)
ax.plot(range(1, 51), mse_val)

# set labels
ax.set_xlabel("K Value")
ax.set_ylabel("Mean Squared Error")
plt.show()