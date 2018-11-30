import pandas as pd
import matplotlib.pyplot as plt

def import_priceData(filename):
    """
    This method creates a time series data frame with price data specified
    :param filename: the name of the file containing the price data (as csv)
    :return: A data frame containing only the date time and open price for the given asset
    """
    price_df = pd.read_csv(filename, encoding='utf-8', header=0)

    price_df = price_df.set_index(pd.to_datetime(price_df['Date'], format='%Y-%m-%d %I-%p'))

    price_df.drop(price_df.columns.difference(['Close']), 1, inplace=True)

    price_df.columns = ['Price']

    return price_df

if __name__ == "__main__":

    price_dataset = import_priceData("/Users/stenzel/Documents/EECE2300/cryptoanalyzer/src/data/raw/Bittrex_BTCUSD_1h.csv")

    price_dataset.to_csv('cleaned_price_data.csv', sep=',')

    # plt.plot(price_dataset)
    #plt.show()
    #plt.close()

    print(price_dataset)