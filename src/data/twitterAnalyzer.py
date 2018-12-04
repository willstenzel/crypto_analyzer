import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


def import_tweets(filename, date_col, text_col):
    """
    This method creates a time series data frame with tweet date time and text
    :param filename: the name of the file containing the tweets (as csv)
    :param date_col: the string name of the column that contains the dates
    :param text_col: the string name of the column that contains the tweet text
    :return: A data frame containing the date and time, the text of the tweet
    """
    # import data from csv file via pandas library
    # tweet_dataset = pd.read_csv(filename, encoding='utf-8', header=0)
    # tweet_dataset.to_pickle("./intermediate/pickled_tweets.pkl")

    tweet_dataset = pd.read_pickle("./intermediate/pickled_tweets.pkl")
    tweet_dataset.sort_values(by=['date'], inplace=True)
    tweet_dataset = tweet_dataset.set_index(pd.to_datetime(tweet_dataset[date_col], unit='s'))

    tweet_dataset.drop(tweet_dataset.columns.difference([text_col]), 1, inplace=True)

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

    print(tweet_dataset)

    tweet_text = list(tweet_dataset['Text'])

    total_tweets = len(tweet_text)

    print("Total to process: " + str(len(tweet_text)))

    for index, text in enumerate(tweet_text):
        if index % 100000 == 0:
            print(str(index / total_tweets * 100) + "% processed so far")
        score = sid.polarity_scores(text)
        sentiment_scores.append(score['compound'])

    tweet_dataset['Sentiment'] = sentiment_scores
    return tweet_dataset


def get_hourly_sentiment(tweet_dataset):
    """
    Computes the mean the sentiment score by hour and normalizes the data
    :param tweet_dataset: the dataframe that we want to find the hourly sentiment score of
    :return: the dataframe with the mean score for the hour
    """

    tweet_dataset = tweet_dataset.resample('H').mean()

    return tweet_dataset


def noramlize_data(df):
    """
    Normalizes the data by subtracting the mean and dividing by the max - min.
    :param df: the dataframe that we are normalizing
    :return: the normalized dataframe
    """

    df_normalized = (df - df.mean()) / (df.max() - df.min())

    return df_normalized

if __name__ == "__main__":
    # tweet_dataset = import_tweets("/Users/stenzel/Documents/EECE2300/cryptoanalyzer/src/data/raw/BitcoinTweets.csv", "date", "text")
    print("tweets imported")
    # tweet_dataset['Text'] = tweet_dataset['Text'].apply(preprocess_tweet)

    tweet_dataset = pd.read_pickle("./intermediate/pickled_tweets_preprocessed.pkl")
    print("tweets preprocessed")
    print(tweet_dataset)
    tweet_dataset = analyze_tweets(tweet_dataset)
    print("tweets analyzed")

    # tweet_dataset = get_hourly_sentiment(tweet_dataset)

    tweet_dataset.to_csv('proccesed_BitcoinTweets.csv', sep=',')

    print(tweet_dataset)
