import argparse
from src.data import twitterAnalyzer, cleanPriceData
from src.visualization import visualize_model

import pandas as pd


parser = argparse.ArgumentParser(description='Predicton based on twitter sentiment analysis.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('twitter_data', type=str, help='csv file contains twitter data.')
parser.add_argument('price_data', type=str, help='csv file contains price history data.')
args = parser.parse_args()
# TODO add flags

twitter_data = args.twitter_data
price_data = args.price_data

# Raw Data preparation
#tweet_df = twitterAnalyzer.import_tweets(twitter_data)
#price_df = cleanPriceData.import_priceData(price_data)

# Data pre-processing and dumping
#tweet_df['Text'] = tweet_df['Text'].apply(twitterAnalyzer.preprocess_tweet)

# Represent data as a model
#tweet_df = twitterAnalyzer.analyze_tweets(tweet_df)
#tweet_df = twitterAnalyzer.get_hourly_sentiment(tweet_df)



## For testing purposes
tweet_df = pd.read_csv("data/Itermediate/twitter-data-with-sentiment.csv", encoding='utf-8')
tweet_df = tweet_df.set_index(pd.DatetimeIndex(tweet_df['Date']))
tweet_df = twitterAnalyzer.get_hourly_sentiment(tweet_df)
price_df = pd.read_csv("data/Itermediate/cleaned_price_data.csv", encoding='utf-8')
price_df = price_df.set_index(pd.DatetimeIndex(price_df['Date']))

# Evaluate model
visualize_model.graph_price_seniment(tweet_df, price_df)

# TODO ???
