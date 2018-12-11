from src.data import pct_change_calculator






offset = range(24)
corr = []
for i in offset:
	print("Offset: " + str(i))
	pct_change_df = pct_change_calculator.pct_change(price_df, "2017-07-01 11:00:00", "2018-03-12 19:00:00", i)
	corr = correlation(tweet_df, pct_change_df, i)
	corr.append(corr)