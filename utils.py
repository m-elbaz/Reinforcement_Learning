import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import yfinance as yf

def create_data(stock):
  stck = yf.Ticker(stock)
  prices = stck.history(start='2015-01-01', interval='1d')[['Open', 'High', 'Low', 'Close']]
  prices.to_csv('data/daily_%s.csv'%stock)

def get_data(stock=['AAPL','NVDA','TSLA']):
  stcks=[]
  for s in stock:
    a = pd.read_csv('data/daily_%s.csv' % s, usecols=['close', "Date"], index_col="Date")
    s_ = preprocess(a).values
    stcks.append(s_)

  c=np.array(stcks)
  # recent price are at top; reverse it

  return np.array(stcks)


def get_rsi_timeseries(prices, n=30):

  deltas = (prices - prices.shift(1)).fillna(0)

  avg_of_gains = deltas[1:n + 1][deltas > 0].sum() / n
  avg_of_losses = -deltas[1:n + 1][deltas < 0].sum() / n

  rsi_series = pd.Series(0.0, deltas.index)

  up = lambda x: x if x > 0 else 0
  down = lambda x: -x if x < 0 else 0
  i = n + 1
  for d in deltas[n + 1:]:
    avg_of_gains = ((avg_of_gains * (n - 1)) + up(d)) / n
    avg_of_losses = ((avg_of_losses * (n - 1)) + down(d)) / n
    if avg_of_losses != 0:
      rs = avg_of_gains / avg_of_losses
      rsi_series[i] = 100 - (100 / (1 + rs))
    else:
      rsi_series[i] = 100
    i += 1

  return rsi_series

def preprocess(data):

  data['returns_month'] = ((data['close'] - data['close'].shift(25)) / data['close'].shift(25))
  data['returns_month'] = data['returns_month']/(np.sqrt(25)*data['returns_month'].ewm(span=60, adjust=False).std())

  data['returns_2month'] = ((data['close'] - data['close'].shift(2 * 25)) / data['close'].shift(2*25))
  data['returns_2month'] =data['returns_2month']/(np.sqrt(25*2)*data['returns_2month'].ewm(span=60, adjust=False).std())

  data['returns_3month'] = ((data['close'] - data['close'].shift(3 * 25)) / data['close'].shift(3*25))
  data['returns_3month']= data['returns_3month']/(np.sqrt(25*3)*data['returns_3month'].ewm(span=60, adjust=False).std())

  data['returns_year'] = (data['close'] - data['close'].shift(252)) / data['close'].shift(252)
  data['returns_year']= data['returns_year']/(np.sqrt(252)*data['returns_year'].ewm(span=60, adjust=False).std())


  exp1 = data['close'].ewm(span=12, adjust=False).mean()
  exp2 = data['close'].ewm(span=26, adjust=False).mean()
  data['macd'] = exp1 - exp2

  data['rsi'] = get_rsi_timeseries(data['close'])

  data = data.dropna()

  return data

def get_scaler(env):
  """ Takes a env and returns a scaler for its observation space """
  low = [0] * (env.n_stock * 8 + 1)

  high = []
  max_price = env.stock_price_history.max(axis=1)[:,0]
  min_price = env.stock_price_history.min(axis=1)[:,0]
  max_rmo = env.stock_price_history.max(axis=1)[:, 1]
  max_r2mo = env.stock_price_history.max(axis=1)[:, 2]
  max_r3mo = env.stock_price_history.max(axis=1)[:, 3]
  max_ryear = env.stock_price_history.max(axis=1)[:, 4]
  max_macd = env.stock_price_history.max(axis=1)[:, 5]
  max_rsi = env.stock_price_history.max(axis=1)[:, 6]

  max_cash = env.init_invest * 3 # 3 is a magic number...
  max_stock_owned = max_cash // min_price

  for i in max_stock_owned:
    high.append(i)
  for i in max_price:
    high.append(i)
  for i in max_rmo:
    high.append(i)
  for i in max_r2mo:
    high.append(i)
  for i in max_r3mo:
    high.append(i)
  for i in max_ryear:
    high.append(i)
  for i in max_macd:
    high.append(i)
  for i in max_rsi:
    high.append(i)

  high.append(max_cash)

  scaler = StandardScaler()
  scaler.fit([low, high])
  return scaler


def maybe_make_dir(directory):
  if not os.path.exists(directory):
    os.makedirs(directory)

if __name__ == '__main__':

  a=get_data(['AAPL', 'NVDA'])
  create_data('GOOGL')

