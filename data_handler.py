import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import yfinance as yf
from utils import preprocess
import datetime as dt
import os


class Stock():
    def __init__(self, ticker):
        self.ticker = ticker
        self.path = 'data/daily_%s.csv' % self.ticker
        if os.path.isfile(self.path):
            self.price_df = pd.read_csv(self.path, usecols=['close', "Date"],parse_dates=['Date'],  index_col="Date")
            self.start_date=self.price_df.index.min()
            self.end_date=self.price_df.index.max()
        else :
            self.load_data()


    def load_data(self):
        stck = yf.Ticker(self.ticker)
        new_prices = stck.history(start='2015-01-01', interval='1d')[['Close']].rename(columns={'Close': 'close'})
        new_prices.index=new_prices.index.map(lambda x : x.date())
        new_prices.to_csv(self.path)
        self.price_df = pd.read_csv(self.path, usecols=['close', "Date"],parse_dates=['Date'],  index_col="Date")
        self.start_date=self.price_df.index.min()
        self.end_date=self.price_df.index.max()


    def get_features(self, start, end):
        #start should be bigger than 2015

        return preprocess(self.price_df).loc[start:end]


class MultiStock():
    def __init__(self, ticker_list):
        self.ticker_list = ticker_list
        self.stock_list=[]
        self.nb_stocks=len(self.ticker_list)
        for ticker in self.ticker_list:
            self.stock_list.append(Stock(ticker))


    def get_all_features(self, start, end):

        all_features=[]
        all_date=[]
        for stock in self.stock_list:
            if stock.end_date<end:
                stock.load_data()
            feat = stock.get_features(start, end)
            all_features.append(feat)
            all_date.append(feat.index)


        all_date = pd.concat(all_features,join='inner', axis=1 ).index
        for i in range(len(all_features)):
            all_features[i]=all_features[i].reindex(all_date)

        return np.array(all_features), all_date


if __name__ == '__main__':
    start=dt.date(2015,1, 1)
    end=dt.date(2018,1,1)
    st=MultiStock(['AAPL', 'GOOGL'])
    feat, date = st.get_all_features(start, end)

    pass