import argparse
from src.data import twitterAnalyzer


parser = argparse.ArgumentParser(description='Predicton based on twitter sentiment analysis.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('twitter_data', type=str, help='csv file contains twitter data.')
parser.add_argument('price_data', type=str, help='csv file contains price history data.')
args = parser.parse_args()
# TODO add flags

twitter_data = args.twitter_data
price_data = args.price_data

# Raw Data preparation
tweet_df = twitterAnalyzer.import_tweets(twitter_data)

# Data pre-processing and dumping
tweet_df['Text'] = tweet_df['Text'].apply(twitterAnalyzer.preprocess_tweet)

# Represent data as a model
tweet_df = twitterAnalyzer.analyze_tweets(tweet_df)
tweet_df = twitterAnalyzer.get_hourly_sentiment(tweet_df)

# Evaluate model
print(tweet_df)
# TODO ???
