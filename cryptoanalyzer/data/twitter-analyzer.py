import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


#TODO

def import_tweets(filename, header=None):
    # import data from csv file via pandas library
    tweet_dataset = pd.read_csv(filename, encoding='utf-8')
    # the column names are based on sentiment140 dataset provided on kaggle
    tweet_dataset.columns = ['Label', 'date', 'Screen Name', 'Full Name', 'text', 'Tweet ID', 'App', 'Followers',
                             'Follows', 'Retweets', 'Favorites', 'Verified', 'User Since', 'Location', 'Bio',
                             'Profile Image', 'Google Maps']
    tweet_dataset['sentiment'] = "0"

    tweet_dataset.drop(tweet_dataset.columns.difference(['sentiment', 'date', 'text']), 1, inplace=True)
    # in sentiment140 dataset, positive = 4, negative = 0; So we change positive to 1
    tweet_dataset.sentiment = tweet_dataset.sentiment.replace(4, 1)
    return tweet_dataset


def preprocess_tweet(tweet):
    # Preprocess the text in a single tweet
    # arguments: tweet = a single tweet in form of string
    # convert the tweet to lower case
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

    :param tweet_dataset:
    :return:
    """
    sid = SentimentIntensityAnalyzer()
    for index, row in tweet_dataset.iterrows():
        text = str(tweet_dataset['text'][row])
        score = sid.polarity_scores(text)
        row['sentiment'] = score['compound']
    return tweet_dataset


def feature_extraction(data, method="tfidf"):
    # arguments: data = all the tweets in the form of array, method = type of feature extracter
    # methods of feature extractions: "tfidf" and "doc2vec"
    if method == "tfidf":
        tfv = TfidfVectorizer(sublinear_tf=True,
                              stop_words="english")  # we need to give proper stopwords list for better performance
        features = tfv.fit_transform(data)
    elif method == "doc2vec":
        None
    else:
        return "Incorrect inputs"
    return features


def train_classifier(features, label, classifier="logistic_regression"):
    # arguments: features = output of feature_extraction(...), label = labels in array form, classifier = type of classifier
    from sklearn.metrics import roc_auc_score  # we will use auc as the evaluation metric
    if classifier == "logistic_regression":  # auc (train data): 0.8780618441250002
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(C=1.)
    elif classifier == "naive_bayes":  # auc (train data): 0.8767891829687501
        from sklearn.naive_bayes import MultinomialNB
        model = MultinomialNB()
    elif classifier == "svm":  # can't use sklearn svm, as way too much of data so way to slow. have to use tensorflow for svm
        from sklearn.svm import SVC
        model = SVC()
    else:
        print("Incorrect selection of classifier")
    # fit model to data
    model.fit(features, label)
    # make prediction on the same (train) data
    probability_to_be_positive = model.predict_proba(features)[:, 1]
    # chcek AUC(Area Undet the Roc Curve) to see how well the score discriminates between negative and positive
    print("auc (train data):", roc_auc_score(label, probability_to_be_positive))
    # print top 10 scores as a sanity check
    print("top 10 scores: ", probability_to_be_positive[:10])


# apply the preprocess function for all the tweets in the dataset
#tweet_dataset = import_tweets("/Users/stenzel/Documents/EECE2300/cryptoanalyzer/cryptoanalyzer/data/raw/bitcoin-twitter.csv")
#tweet_dataset['text'] = tweet_dataset['text'].apply(preprocess_tweet)
#tweet_dataset = analyze_tweets(tweet_dataset)

# data = np.array(tweet_dataset.text)
# label = np.array(tweet_dataset.sentiment)
# features = feature_extraction(data, method="tfidf")
# train_classifier(features, label, "logistic_regression")

tweet_dataset = pd.read_csv("twitter-data-with-sentiment.csv", encoding='utf-8')

tweet_dataset.to_csv('twitter-data-with-sentiment.csv', sep=',')

plt.plot(tweet_dataset)
print(tweet_dataset)