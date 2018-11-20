import nltk
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

sid = SentimentIntensityAnalyzer()

sentiment = sid.polarity_scores("RT @michaelkohler20: Why Bitcoin Will Fail - Why There Will be a bitcoincrash 100% And I Will Not Invest In It! https://t.co/UvR4KlERHb")

print(sentiment)

