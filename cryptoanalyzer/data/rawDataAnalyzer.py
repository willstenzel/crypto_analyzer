import pandas as pd
import matplotlib.pyplot as plt


if __name__ == "__main__":
    path_in = '/Users/stenzel/Documents/EECE2300/cryptoanalyzer/cryptoanalyzer/data/raw/Bittrex_BTCUSD_1h.csv'

    sample_data = pd.read_csv(path_in, skiprows=1)
    print(sample_data.head())

    #df = pd.DataFrame([sub.split(",") for sub in sample_data])

    #print(df.head())

    #plt.plot(sample_data)

    sample_data.plot(kind="line", y="Close")