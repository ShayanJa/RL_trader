import os
from binance.client import Client
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
import time
from plotly import tools
import click

PRICE_PATH = "asset_prices/"
PLOT_PATH = "asset_prices/plots/"

# Initialize Binance client
api_key = os.getenv("CLIENT_KEY")
api_secret = os.getenv("SECRET_KEY")
client = Client(api_key, api_secret)

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

@click.group()
def main():
  pass

@main.command()
@click.argument('crypto')
def price(crypto):
  klines = client.get_historical_klines(crypto, Client.KLINE_INTERVAL_1MINUTE, "100 day ago UTC")
  prices = pd.DataFrame(klines, columns=_columns).astype(float)

  # Change timestamp to datetime
  prices['Open time'] = prices['Open time'].apply(lambda x: dt.datetime.fromtimestamp(int(x)/1000))
  prices = prices.set_index('Open time')
  # Save data
  prices.to_csv('btcusd-min.csv')

  # Display data
  plt.plot(prices['Close'])
  plt.xlabel('Date')
  plt.ylabel('Price')
  plt.savefig(PLOT_PATH + crypto + ".png")

  plt.show()


if __name__ == '__main__':
  main()