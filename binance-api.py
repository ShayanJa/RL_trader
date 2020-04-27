import os
from binance.client import Client
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
import time

api_key = os.getenv("CLIENT_KEY")
api_secret = os.getenv("SECRET_KEY")

_columns=[
  'Open time',
  'Open',
  'High',
  'Low',
  'Close',
  'Volume',
  'Close time',
  'Quote asset volume',
  'Number of trades',
  'Taker buy base asset volume',
  'Taker buy quote asset volume',
  'ignore'
]

client = Client(api_key, api_secret)

klines = client.get_historical_klines("BTCUSDT", Client.KLINE_INTERVAL_1DAY, "July 4, 2013 PST")

df = pd.DataFrame(klines, columns=_columns).astype(float)
# Change timestamp to datetime
df['Open time'] = df['Open time'].apply(lambda x: dt.datetime.fromtimestamp(int(x)/1000))

# Plot binance
df.plot(kind='line', x='Open time', y='Close')
print(df['Open time'])
plt.show()

