import ta
import pandas as pd
import datetime as dt
import numpy as np

CSV_PATH = 'btcusd-min.csv'

class Features():
  def __init__(self, price_data, feature_length):
    self.price_data = price_data
    self.feature_length = feature_length
    self.bolingerbands = ta.volatility.BollingerBands(close=price_data["Close"])
    self.rsi = ta.momentum.RSIIndicator(close=price_data['Close'])
    self.ao = ta.momentum.AwesomeOscillatorIndicator(high=price_data['High'], low=price_data['Low'])
    self.macd = ta.trend.MACD(close=price_data['Close'], fillna=True)
    self.mass_index = ta.trend.MassIndex(high=price_data['High'],low=price_data['Low'])

    self.features = pd.DataFrame({
      "daily_returns": self.price_data['Close'].pct_change().fillna(0)*100,
      "rsi": self.rsi.rsi(), 
      "macd": self.macd.macd(), 
      "BBH": self.bolingerbands.bollinger_hband(),
      "BBL": self.bolingerbands.bollinger_lband(),
      "BBM": self.bolingerbands.bollinger_mavg(),
      "MI":  self.mass_index.mass_index()
    })
    # self.features = self.features.iloc[:feature_length]
    self.price_data = self.price_data.iloc[feature_length:]

    self.max_drawdown = .25
    self.position_increment = .005
  
  def getFeaturesAsList(self, n:int):
    return self.features.iloc[n:, :]


date_split = dt.datetime(2020, 3, 14, 1, 0)  # @param

# Split data into training and test set
prices = pd.read_csv(CSV_PATH, parse_dates=True, index_col=0)
train = prices[:date_split]
test = prices[date_split:]

s = Features(prices, 10)
print(s.features[:date_split])
# print(len(s.features.columns))

# print(s.getFeaturesAsList(10))



