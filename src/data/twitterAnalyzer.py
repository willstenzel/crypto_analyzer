import pandas as pd
import re
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


def import_tweets(filename):
    """
    This method creates a time series data frame with tweet date time and text
    :param filename: the name of the file containing the tweets (as csv)
    :return: A data frame containing the date and time, the text of the tweet
    """
    # import data from csv file via pandas library
    tweet_dataset = pd.read_csv(filename, encoding='utf-8', header=0)
    # the column names are based on sentiment140 dataset provided on kaggle

    tweet_dataset = tweet_dataset.set_index(pd.DatetimeIndex(tweet_dataset['Date']))

    tweet_dataset.drop(tweet_dataset.columns.difference(['Tweet Text']), 1, inplace=True)

    tweet_dataset.columns = ['Text']

    return tweet_dataset


def preprocess_tweet(tweet):
    """
    Preprocesses the text in a single tweet
    :param tweet: a single tweet in form of string
    :return: the lowercase text with the unnecessary characters removed
    """
    tweet.lower()
    # convert all urls to sting "URL"
    tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', 'URL', tweet)
    # convert all @username to "AT_USER"
    tweet = re.sub('@[^\s]+', 'AT_USER', tweet)
    # correct all multiple white spaces to a single white space
    tweet = re.sub('[\s]+', ' ', tweet)
    # convert "#topic" to just "topic"
    tweet = re.sub(r'#([^\s]+)', r'\1', tweet)
    return tweet


def analyze_tweets(tweet_dataset):
    """
    Adds then sentiment score to each tweet to the dataframe
    :param tweet_dataset: the dataframe that we want to add the sentiment score to
    :return: the dataframe with the correct sentiment scores
    """
    sid = SentimentIntensityAnalyzer()
    sentiment_scores = []
    for index, row in tweet_dataset.iterrows():
        text = str(row['Text'])
        score = sid.polarity_scores(text)
        sentiment_scores.append(score['compound'])
    tweet_dataset['Sentiment'] = sentiment_scores
    return tweet_dataset

def get_hourly_sentiment(tweet_dataset):
    """
    Sums the sentiment score by hour
    :param tweet_dataset: the dataframe that we want to find the hourly sentiment score of
    :return: the dataframe with the mean score for the hour
    """

    return tweet_dataset.resample('H').mean()







if __name__ == "__main__":
    # apply the preprocess function for all the tweets in the dataset
    # tweet_dataset = import_tweets("/Users/stenzel/Documents/EECE2300/cryptoanalyzer/src/data/raw/bitcoin-twitter.csv")
    # tweet_dataset['Text'] = tweet_dataset['Text'].apply(preprocess_tweet)
    # tweet_dataset = analyze_tweets(tweet_dataset)

    #tweet_dataset.to_csv('twitter-data-with-sentiment.csv', sep=',')

    tweet_dataset = pd.read_csv("Itermediate/twitter-data-with-sentiment.csv", encoding='utf-8')

    tweet_dataset = tweet_dataset.set_index(pd.DatetimeIndex(tweet_dataset['date']))
    tweet_dataset = get_hourly_sentiment(tweet_dataset)

    plt.plot(tweet_dataset)

    print(tweet_dataset)