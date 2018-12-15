import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# earliest possible start and end dates in both datasets: "2017-07-01 11:00:00", "2018-03-12 19:00:00"

def pct_change(df, start, end, hours_offset):
    px_change_li = []
    t = 0
    for px_at_t in df.loc[start:end].iloc[:, 0]:
        if t < hours_offset:
            pass
        else:
            px_change = px_at_t / df.loc[start:end].iloc[t-hours_offset, 0] - 1
            px_change_li.append(px_change)
        t = t + 1
    px_change_ar = np.array(px_change_li)
    pct_change_df = pd.DataFrame(index=pd.DatetimeIndex(start=start, end=end, freq="H"), columns=["% Change"])
    pct_change_df.drop(pct_change_df.index[0:hours_offset], inplace=True)
    print(len(pct_change_df))
    print(len(px_change_ar))
    pct_change_df["% Change"] = px_change_ar * 100

    return pct_change_df


def barplot(sentiments, pct_changes, hours_offset):

    x = pct_changes.index #times, for each pair of bars

    ind = np.arange(len(x[hours_offset:])) #adjust for offset making fewer data points

    sentiments = sentiments.iloc[0:ind[-1]+1, 0] * 10
    pct_changes = pct_changes.iloc[hours_offset:, 0] #offset pct changes to they come after corresponding sentiments
    width = .3

    fig, ax = plt.subplots()
    splot = ax.bar(ind+width/2, sentiments, width, color="r", align="edge")

    pct_changes = pct_changes
    pcplot = ax.bar(ind+width/2, pct_changes, -width, color="b", align="edge")

    ax.set_ylabel("Sentiment, Percent Change")
    if hours_offset == 1:
        ax.set_title("Twitter Sentiment and Percent Change in Bitcoin Price Over the Next Hour")
    else:
        ax.set_title("Twitter Sentiment and Percent Change in Bitcoin Price Over the Next "
                     + str(hours_offset) + " Hours")
    # ax.set_xticks(ind + width / 2)
    # ax.set_xticklabels(x)
    ax.set_xlabel("Hourly data From " + str(x[0]) + " to " + str(x[-1]))
    ax.legend((splot[0], pcplot[0]), ("Sentiment", "% Change"))
    ax.autoscale_view()

    #show every 4th x
    # l = len(x) / 4
    # num_labels = int(round(l))
    # plt.xticks(i * 4 for i in range(num_labels))

    plt.show()

    print("The correlation is " + str(np.corrcoef(sentiments, pct_changes)[0, 1]))


def splot(sentiments, pct_changes, hours_offset):

    x = pct_changes.index  # times, for each pair of bars

    ind = np.arange(len(x[hours_offset:]))  # adjust for offset making fewer data points

    sentiments = sentiments.iloc[0:ind[-1] + 1, 0]
    pct_changes = pct_changes.iloc[hours_offset:, 0]

    plt.title("Percent Change vs. Sentiment")
    plt.xlabel("Sentiment Scores")
    if hours_offset == 1:
        plt.ylabel("Percent Change in Price over the next Hour")
    else:
        plt.ylabel("Percent Change in Price over the next " + str(hours_offset) + " Hours")

    plt.scatter(sentiments, pct_changes, s=5)
    plt.show()


def correlation(sentiments, pct_changes, hours_offset):

    x = pct_changes.index  # times, for each pair of bars
    ind = np.arange(len(x[hours_offset:]))  # adjust for offset making fewer data points

    sentiments = sentiments.iloc[0:ind[-1]+1, 0]
    pct_changes = pct_changes.iloc[hours_offset:, 0] #offset pct changes to they come after corresponding sentiments

    corr = np.corrcoef(sentiments, pct_changes)

    print("The correlation is " + str(corr[0, 1]))

    return corr


def mean(data):
    """Return the sample arithmetic mean of data."""
    n = len(data)
    if n < 1:
        raise ValueError('mean requires at least one data point')
    return sum(data)/n # in Python 2 use sum(data)/float(n)

def _ss(data):
    """Return sum of square deviations of sequence data."""
    c = mean(data)
    ss = sum((x-c)**2 for x in data)
    return ss

def stddev(data, ddof=0):
    """Calculates the population standard deviation
    by default; specify ddof=1 to compute the sample
    standard deviation."""
    n = len(data)
    if n < 2:
        raise ValueError('variance requires at least two data points')
    ss = _ss(data)
    pvar = ss/(n-ddof)
    return pvar**0.5


if __name__ == "__main__":

    # tweet_df = pd.read_pickle("data/intermediate/proccesed_BitcoinTweets.pkl")
    # tweet_df = tweet_df.set_index(pd.DatetimeIndex(tweet_df['date']))
    # tweet_df = twitterAnalyzer.get_hourly_sentiment(tweet_df)
    # tweet_df = twitterAnalyzer.noramlize_data(tweet_df)
    tweet_df = pd.read_pickle("/Users/maxhopley/Documents/EECE2300/crypto_analyzer/src/data/intermediate/proccesed_normalized_BitcoinTweets.pkl")
    # print("2017-12-12 15:00:00" in tweet_df.index)

    price_df = pd.read_pickle("/Users/maxhopley/Documents/EECE2300/crypto_analyzer/src/data/intermediate/cleaned_price_data.pkl")
    price_df = price_df.set_index(pd.DatetimeIndex(price_df['Date']))
    price_df = price_df.drop('Date', 1)
    price_df = price_df.iloc[::-1]
    pct_change_df = pct_change(price_df, "2017-07-01 11:00:00", "2018-03-12 19:00:00", 1)

    splot(tweet_df, pct_change_df, 1)
