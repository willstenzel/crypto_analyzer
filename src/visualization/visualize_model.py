import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def graph_price_seniment(sentiment, price):
    """
    This method graphs the sentiment as a bar graph over the price which is a line graph
    :param sentiment: the sentiment dateTime series containing sentiment scores at a given time
    :param price: the price dateTime series which has the value for each hour
    """

    sentiment = pd.Series(sentiment['Sentiment'], index=sentiment.index)
    price = pd.Series(price['Price'], index=price.index)

    # reverse the price data so that it is in ascending order like the sentiment data
    price = price.iloc[::-1]

    # keep rows that have the share the same date times
    price = price[sentiment.index[0]:sentiment.index[-1]]

    # scale the sentiment to better match the price values

    # create a plot of the sentiment as a bar plot and the price as a line plot

    # Plot graph with 2 y axes
    fig, ax1 = plt.subplots()

    # Plot Line
    ax1.plot(price.index, price.values, 'r-')
    # Set the x-axis label
    ax1.set_xlabel('Time')

    # Set the y-axis label
    ax1.set_ylabel('Price')

    # Set up ax2 to be the second y axis with x shared
    ax2 = ax1.twinx()
    # Plot a Bar
    ax2.bar(sentiment.index, sentiment, width=5, alpha=0.3)


    plt.show()



if __name__ == "__main__":

    twitter_data = 'data/raw/bitcoin-twitter.csv'

    price_data = 'data/raw/Bittrex_BTCUSD_1h.csv'
