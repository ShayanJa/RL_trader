import ta
import pandas as pd
import datetime as dt
import numpy as np

class Features():
  """
  Features will hold time series data relevant to technical analysis

  Requires:
  price_data - A Dataframe of OHLC bars
  feature_length - The lookback period for all the features"""

  def __init__(self, price_data, feature_length):
    self.price_data = price_data
    self.feature_length = feature_length
    self.rsi = ta.momentum.RSIIndicator(close=price_data['Close'])
    self.macd = ta.trend.MACD(close=price_data['Close'], fillna=True)

    self.features = pd.DataFrame({
      "daily_returns": self.price_data['Close'].pct_change().fillna(0)*100,
      "rsi": self.rsi.rsi(), 
      "macd": self.macd.macd()
    })

    # We will be shifting the start date to the length of feature_length
    # because we want to initialize our features and
    # agent state with nonzero data. This will help when learning
    # on smaller amounts of data 
    self.price_data = self.price_data.iloc[feature_length:]



if __name__ == "__main__":
  
  CSV_PATH = 'asset_prices/btcusd.csv'

  # Split data into training and test set
  date_split = dt.datetime(2020, 3, 14, 1, 0)  # @param
  prices = pd.read_csv(CSV_PATH, parse_dates=True, index_col=0)
  train = prices[:date_split]
  test = prices[date_split:]

  s = Features(prices, 10)
  print(s.features[:date_split])



