import matplotlib.pyplot as plt
import datetime as dt
import time
from plotly.graph_objs import *
import pandas_datareader.data as web
import click

START = dt.datetime(2015, 1, 1)
END = dt.datetime(2020, 4, 27)

PRICE_PATH = "asset_prices/"
PLOT_PATH = "plots/"

@click.group()
def main():
  pass

@main.command()
@click.argument('stock')
def price(stock):
  # Save price
  prices  = web.DataReader(stock, 'yahoo', START, END)
  prices.to_csv(PRICE_PATH + stock + '.csv')

  # Display data
  plt.plot(prices['Close'])
  plt.xlabel('Date')
  plt.ylabel('Price')
  plt.savefig(PLOT_PATH + stock + ".png")

  plt.show()


if __name__ == '__main__':
  main()
