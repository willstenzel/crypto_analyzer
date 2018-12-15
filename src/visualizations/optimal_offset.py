import pandas as pd
import matplotlib.pyplot as plt
from src.data import pct_change_calculator


tweet_df = pd.read_pickle("/Users/stenzel/Documents/EECE2300/crypto_analyzer/src/data/intermediate/proccesed_normalized_BitcoinTweets.pkl")
price_df = pd.read_pickle("/Users/stenzel/Documents/EECE2300/crypto_analyzer/src/data/intermediate/cleaned_price_data.pkl")
price_df = price_df.set_index(pd.DatetimeIndex(price_df['Date']))
price_df = price_df.drop('Date', 1)
price_df = price_df.iloc[::-1]


offset = range(0, 150, 5)
correlations = []
for i in offset:
	print("Offset: " + str(i))
	pct_change_df = pct_change_calculator.pct_change(price_df, "2017-07-01 11:00:00", "2018-03-12 19:00:00", i)
	corr = pct_change_calculator.correlation(tweet_df, pct_change_df, i)
	correlations.append(corr[0, 1])


print(correlations)
# initialize a new figure
fig, ax = plt.subplots()

plt.title('Correlation Over Different Hour Offsets', y=1.00)
ax.plot(offset, correlations)

# set labels
ax.set_xlabel("Hours Offset")
ax.set_ylabel("Correlation")
plt.show()