#dataPull module

import numpy as np
import pandas as pd
from pandas_datareader import data
import talib

def getCoinData(symbol, start, end):
    """Param:
        symbol: string type, yahoo finance ticker for coin
        start: string type, start date for pull (yyyy/mm/dd)
        end: string type, end date for pull (yyyy/mm/dd)

       Returns: Dataframe of Open, High, Low, Close, Adj. Close and Volume
                with index representing each trading day"""
    
    coin_data = data.get_data_yahoo(symbol, start, end)
    coin_data = coin_data.rename(columns={'Adj Close': 'Adj. Close'})

    return coin_data

def removeDifferentDates(df1, df2):
    """Param:
        df1: Dataframe type, first dataframe
        df2: Dataframe type, second dataframe

       Returns: list object of length 2, with modified df1 and df2 in that
       order, with date indexes exactly the same"""
    dates1 = list(df1.index)
    dates2 = list(df2.index)
    
    i = 0

    while i < len(dates2):
        dates2[i] = dates2[i]
        i = i + 1

    index = 0

    while index < len(dates1):
        dates1[index] = dates1[index]
        index = index + 1

    index = 0
    
    df1.index = dates1
    df2.index = dates2
    
    overallIndexes = list(set(dates1).intersection(set(dates2)))

    checkIndex = 0

    df1 = df1[df1.index.isin(overallIndexes)]
    df2 = df2[df2.index.isin(overallIndexes)]

    return [df1, df2]

def generateTechnicalDataframe(coin_dataframe):
    """Param:
    coin_dataframe: Dataframe type, exact same format as output of getCoinData

    Returns: dataframe of technical data, with index representing dates and
             columns representing each technical indicator"""
    
    Low_df = coin_dataframe['Low']
    High_df = coin_dataframe['High']
    Price_df = coin_dataframe['Adj. Close']
    
    LowList = Low_df.values.tolist()
    HighList = High_df.values.tolist()
    PriceList = Price_df.values.tolist()
    DateList = coin_dataframe.index.tolist()

    #converted to list

    z = 0

    while z < len(LowList):
        if isinstance(LowList[z], str):
            LowList[z] = float(LowList[z].replace(',', ''))
        z = z + 1

    y = 0

    while y < len(HighList):
        if isinstance(HighList[y], str):
            HighList[y] = float(HighList[y].replace(',', ''))
        y = y + 1

    x = 0

    while x < len(PriceList):
        if isinstance(PriceList[x], str):
            PriceList[x] = float(PriceList[x].replace(',', ''))
        x = x + 1

    #type conversions complete, string --> float

    Low = np.array(LowList)
    High = np.array(HighList)
    Close = np.array(PriceList)

    #Low, High, and Close converted to Array format (TA-Lib calls require Array)

    SARtoList = (talib.SAR(High, Low, acceleration = 0.2, maximum = 0.20))
    BBandsArray = (talib.BBANDS(Close, timeperiod = 5, nbdevup = 2, nbdevdn = 2, matype = 0))
    EMAList = talib.EMA(Close, timeperiod=30)
    KAMAList = talib.KAMA(Close, timeperiod=30)
    MAList = talib.MA(Close, timeperiod=30, matype=0)
    WMAList = talib.WMA(Close, timeperiod=30)
    TRIMAList = talib.TRIMA(Close, timeperiod=30)
    TEMAList = talib.TEMA(Close, timeperiod=30)
    HTList = talib.HT_TRENDLINE(Close)
    ADXList = talib.ADX(High, Low, Close, timeperiod=14)
    ADXRList = talib.ADXR(High, Low, Close, timeperiod=14)
    CMOList = talib.CMO(Close, timeperiod=14)
    DXList = talib.DX(High, Low, Close, timeperiod=14)
    MACDArray = talib.MACDFIX(Close, signalperiod=9)
    MINUS_DI_List = talib.MINUS_DI(High, Low, Close, timeperiod=14)
    PLUS_DI_List = talib.PLUS_DI(High, Low, Close, timeperiod=14)
    MOMList = talib.MOM(Close, timeperiod=10)
    RSIList = talib.RSI(Close, timeperiod=14)
    NATRList = talib.NATR(High, Low, Close, timeperiod=14)
    BETAList = talib.BETA(High, Low, timeperiod=5)
    ROCList = talib.ROC(Close, timeperiod=10)
    WILLRList = talib.WILLR(High, Low, Close, timeperiod=14)
    ULTOSCList = talib.ULTOSC(High, Low, Close, timeperiod1=7, timeperiod2=14, timeperiod3=28)


    #method calls to TA-Lib complete, results stored in SARtoList (list) and BBandsArray (array)

    toCombine = []

    BBandsUpperDF = pd.DataFrame(BBandsArray[0], columns = ['Upper Band',])
    toCombine.append(BBandsUpperDF)

    BBandsMiddleDF = pd.DataFrame(BBandsArray[1], columns = ['Middle Band',])
    toCombine.append(BBandsMiddleDF)

    BBandsLowerDF = pd.DataFrame(BBandsArray[2], columns = ['Lower Band',])
    toCombine.append(BBandsLowerDF)

    MACD_df = pd.DataFrame(MACDArray[0], columns = ['MACD',])
    toCombine.append(MACD_df)

    MACD_Hist_df = pd.DataFrame(MACDArray[1], columns = ['MACD_Hist',])
    toCombine.append(MACD_Hist_df)

    MACD_Sig_df = pd.DataFrame(MACDArray[2], columns = ['MACD_Sig',])
    toCombine.append(MACD_Sig_df)

    DateDF = pd.DataFrame({'Date': DateList,})
    toCombine.append(DateDF)

    SARdf = pd.DataFrame({'SAR': SARtoList,})
    toCombine.append(SARdf)

    EMAdf = pd.DataFrame({'EMA': EMAList,})
    toCombine.append(EMAdf)

    KAMAdf = pd.DataFrame({'KAMA': KAMAList,})
    toCombine.append(KAMAdf)

    MAdf = pd.DataFrame({'MA': MAList,})
    toCombine.append(MAdf)

    WMAdf = pd.DataFrame({'WMA': WMAList,})
    toCombine.append(WMAdf)

    TRIMAdf = pd.DataFrame({'TRIMA': TRIMAList,})
    toCombine.append(TRIMAdf)

    TEMAdf = pd.DataFrame({'TEMA': TEMAList,})
    toCombine.append(TEMAdf)

    HTdf = pd.DataFrame({'HT Trendline': HTList,})
    toCombine.append(HTdf)

    ADXdf = pd.DataFrame({'ADX': ADXList,})
    toCombine.append(ADXdf)

    ADXRdf = pd.DataFrame({'ADXR': ADXRList,})
    toCombine.append(ADXdf)

    CMOdf = pd.DataFrame({'CMO': CMOList,})
    toCombine.append(CMOdf)

    MINUSDI_df = pd.DataFrame({'MINUSDI': MINUS_DI_List,})
    toCombine.append(MINUSDI_df)

    PLUSDI_df = pd.DataFrame({'PLUSDI': PLUS_DI_List,})
    toCombine.append(PLUSDI_df)

    MOMdf = pd.DataFrame({'MOM': MOMList,})
    toCombine.append(MOMdf)

    RSIdf = pd.DataFrame({'RSI': RSIList,})
    toCombine.append(RSIdf)

    NATRdf = pd.DataFrame({'NATR': NATRList,})
    toCombine.append(NATRdf)

    BETAdf = pd.DataFrame({'BETA': BETAList,})
    toCombine.append(BETAdf)

    ROCdf = pd.DataFrame({'ROC': ROCList,})
    toCombine.append(ROCdf)

    WILLRdf = pd.DataFrame({'WILLR': WILLRList,})
    toCombine.append(WILLRdf)

    ULTOSCdf = pd.DataFrame({'ULTOSC': ULTOSCList,})
    toCombine.append(ULTOSCdf)

    #All data converted to DataFrame type

    TA_df = pd.concat(toCombine, axis = 1,)

    TA_df = TA_df.set_index('Date')

    return TA_df

def generateLabels(coin_dataframe, days):
    listUpDown = []
    DateList = []

    closePricesDF = coin_dataframe['Adj. Close']
    
    listOfDates = closePricesDF.index.tolist()
    listOfPrices = closePricesDF.tolist()

    index = 0

    later = days - 1

    while later < len(listOfPrices):
        if listOfPrices[later] > listOfPrices[index]:
            listUpDown.append(1)
            DateList.append(listOfDates[index])
        else:
            listUpDown.append(0)
            DateList.append(listOfDates[index])
        index = index + 1
        later = later + 1

    UpDownDF = pd.DataFrame({'Date': DateList, 'Up/Down': listUpDown})

    UpDownDF = UpDownDF.set_index('Date')

    return UpDownDF
    
    
