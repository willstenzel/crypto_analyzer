import pandas as pd
import numpy as np
from src.visualization import visualize_model
from src.data import twitterAnalyzer
import matplotlib.pyplot as plt


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
    pct_change_df["% Change"] = px_change_ar * 100

    return pct_change_df


def barplot(sentiments, pct_changes, hours_offset):

    x = pct_changes.index #times, for each pair of bars

    ind = np.arange(len(x[hours_offset:])) #adjust for offset making fewer data points

    sentiments = sentiments.iloc[0:ind[-1]+1, 0]/100
    pct_changes = pct_changes.iloc[hours_offset:, 0] #offset pct changes to they come after corresponding sentiments
    width = .3

    fig, ax = plt.subplots()
    splot = ax.bar(ind+width/2, sentiments, width, color="r", align="edge")

    pct_changes = pct_changes
    pcplot = ax.bar(ind+width/2, pct_changes, -width, color="b", align="edge")

    ax.set_ylabel("must set two y labels")
    ax.set_title("title")
    ax.set_xticks(ind + width / 2)
    ax.set_xticklabels(x)
    ax.legend((splot[0], pcplot[0]), ("Sentiment/100", "% Change"))
    ax.autoscale_view()
    print("The correlation is " + str(np.corrcoef(sentiments, pct_changes)[0, 1]))

    plt.show()


if __name__ == "__main__":
    ## For testing purposes
    tweet_df = pd.read_csv("Itermediate/twitter-data-with-sentiment.csv", encoding='utf-8')
    tweet_df = tweet_df.set_index(pd.DatetimeIndex(tweet_df['Date']))
    tweet_df = twitterAnalyzer.get_hourly_sentiment(tweet_df)
    price_df = pd.read_csv("Itermediate/cleaned_price_data.csv", encoding='utf-8')
    price_df = price_df.set_index(pd.DatetimeIndex(price_df['Date']))
    price_df = price_df.drop(columns="Date")
    price_df = price_df.iloc[::-1]
    #print(price_df[:9])
    #print(tweet_df)
    pct_change_df = pct_change(price_df, "2017-12-12 15:00:00", "2017-12-12 19:00:00", 1)
    #print(pct_change_df)
    barplot(tweet_df, pct_change_df, 1)

