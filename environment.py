import abc
import tensorflow as tf
import numpy as np
import ta
import os
import pandas as pd
from binance.client import Client
import datetime as dt

from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils
from tf_agents.specs import array_spec
from tf_agents.environments import wrappers
from tf_agents.trajectories import time_step as ts

tf.compat.v1.enable_v2_behavior()

class TradingEnvironment(py_environment.PyEnvironment):
  def __init__(self, initial_balance, price_data, price_history_t, mean_history_t, macd_t, fast_ema, slow_ema):
    self.t = 0
    self.fast_ema = fast_ema
    self.slow_ema = slow_ema
    self.price_data = price_data
    self.price_history_t = price_history_t
    self.mean_history_t = mean_history_t
    self.macd_t = macd_t
    self.initial_balance = initial_balance
    self.balance = initial_balance
    self.cash_balance = initial_balance
    self.position_increment = .001
    self.fees = .001
    self.max_drawdown = initial_balance
    self.positions = []
    self._action_spec = array_spec.BoundedArraySpec(
      shape=(), dtype=np.int32, minimum=0, maximum=2, name='action')
    self._observation_spec = array_spec.BoundedArraySpec(
      shape=(price_history_t+macd_t+1,), dtype=np.float32, name="observation")
    self.return_history = [self.price_data.iloc[self.t+1, :]['Close'] - self.price_data.iloc[self.t, :]['Close']  for _ in range(self.price_history_t)]
    self.mean_data = self.price_data.rolling(20, min_periods=1).mean()
    # self.mean_history = [self.mean_data.iloc[self.t-k, :]['Close'] for k in reversed(range(self.mean_history_t))]
    self.MACD_trend = ta.trend.ema_indicator(self.price_data['Close'], self.fast_ema) - ta.trend.ema_indicator(self.price_data['Close'], self.slow_ema)
    self.MACD_trend = self.MACD_trend.fillna(self.MACD_trend[slow_ema]).tolist()
    self.MACD = [self.MACD_trend[self.t] for _ in reversed(range(self.macd_t))]
    self._episode_ended = False

  def observation_spec(self):
    return self._observation_spec
  
  def _step(self, action):
    if self._episode_ended:
      return self.reset()
    
    # action = 0: Hold, 1: Buy, 2: Sell
    rewards = 0
    if action == 0:
      unrealized_profits = 0
      for p in self.positions:
        unrealized_profits += (self.price_data.iloc[self.t, :]['Close'] - p) * self.position_increment *(1-self.fees)
      # rewards = unrealized_profits + sum(self.mean_history)/self.mean_history_t
      rewards = unrealized_profits

    elif action == 1:
      p = self.price_data.iloc[self.t, :]['Close'] * self.position_increment * (1+self.fees)
      if p > self.cash_balance:
        rewards = -1
        # pass
      else:
        self.cash_balance -= p 
        self.positions.append(self.price_data.iloc[self.t, :]['Close'])
        rewards += self.MACD[-1] - p
    elif action == 2:
      if len(self.positions) == 0:
        rewards = -1
      else:
        p = self.positions.pop(0)
        profits = (self.price_data.iloc[self.t, :]['Close'] - p) * self.position_increment *(1-self.fees)
        self.cash_balance += (self.price_data.iloc[self.t, :]['Close']* self.position_increment)*(1-self.fees)
        rewards = profits

    self.balance = self.cash_balance
    for p in self.positions:
      self.balance += self.price_data.iloc[self.t, :]['Close'] * self.position_increment
    
    self.max_drawdown = min(self.max_drawdown, self.balance)
    self.t += 1
    self.return_history.pop(0)
    self.return_history.append(self.price_data.iloc[self.t, :]['Close'] - self.price_data.iloc[self.t-1, :]['Close'])
    self.MACD.pop(0)
    self.MACD.append(self.MACD_trend[self.t])
    if self.t == len(self.price_data)-1:
      self._episode_ended = True

    self._state = [self.balance] + self.return_history + self.MACD
    
    return ts.transition(
      np.array(self._state, dtype=np.float32), reward=rewards, discount=.9)
   
  def action_spec(self):
    return self._action_spec
  
  def _reset(self):
    self.t = 0
    self._episode_ended = False
    self.profits = 0
    self.balance = self.initial_balance
    self.cash_balance = self.initial_balance
    self.positions = []
    self.return_history = [self.price_data.iloc[self.t+1, :]['Close'] - self.price_data.iloc[self.t, :]['Close']  for _ in range(self.price_history_t)]
    self.mean_history = [self.mean_data.iloc[self.t-k, :]['Close'] for k in reversed(range(self.mean_history_t))]
    self.MACD = [self.MACD_trend[self.t] for _ in reversed(range(self.macd_t))]
    self._state = [self.balance] + self.return_history + self.MACD
    return ts.restart(np.array(self._state, dtype=np.float32))

  def buy_and_hold(self):
    amount =  self.initial_balance / self.price_data.iloc[0, :]['Close']
    return self.price_data * amount
  
  def get_max_drawdown(self):
    return (self.max_drawdown-self.initial_balance)/self.initial_balance

  
class LiveBinanceEnvironment(py_environment.PyEnvironment):
  def __init__(self, asset1, asset2, position_increment, price_history_t, mean_history_t, macd_t, fast_ema, slow_ema):
    self.asset1 = asset1
    self.asset2 = asset2
    self.assetpair = asset1+asset2
    self.position_increment = position_increment
    self.fast_ema = fast_ema
    self.slow_ema = slow_ema
    self.price_history_t = price_history_t
    self.mean_history_t = mean_history_t
    self.macd_t = macd_t

    # Initialize Binance client
    api_key = os.getenv("CLIENT_KEY")
    api_secret = os.getenv("SECRET_KEY")
    self.client = Client(api_key, api_secret)
    self._columns=[
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

    # get price_data for 1 month ago
    prices = self.client.get_historical_klines(self.assetpair, self.client.KLINE_INTERVAL_1MINUTE, "1 day ago UTC")
    prices = pd.DataFrame(prices, columns=self._columns).astype(float)

    # Change timestamp to datetime
    prices['Open time'] = prices['Open time'].apply(lambda x: dt.datetime.fromtimestamp(int(x)/1000))
    self.price_data = prices.set_index('Open time')

    # Get Account balance
    self.initial_balance = self.client.get_asset_balance(asset='USDT')['free']
    self.balance = self.initial_balance

    # Setup state
    self.return_history = [self.price_data.iloc[-k, :]['Close'] - self.price_data.iloc[-k-1, :]['Close']  for k in reversed(range(self.price_history_t))]
    self.mean_data = self.price_data.rolling(20, min_periods=1).mean()
    # self.mean_history = [self.mean_data.iloc[self.t-k, :]['Close'] for k in reversed(range(self.mean_history_t))]
    self.MACD_trend = ta.trend.ema_indicator(self.price_data['Close'], self.fast_ema) - ta.trend.ema_indicator(self.price_data['Close'], self.slow_ema)
    self.MACD_trend = self.MACD_trend.fillna(self.MACD_trend[slow_ema]).tolist()
    self.MACD = [self.MACD_trend[-k] for k in reversed(range(self.macd_t))]
    self._action_spec = array_spec.BoundedArraySpec(
      shape=(), dtype=np.int32, minimum=0, maximum=2, name='action')
    self._observation_spec = array_spec.BoundedArraySpec(
      shape=(price_history_t+macd_t+1,), dtype=np.float32, name="observation")
    self.orders = []

  def observation_spec(self):
    return self._observation_spec

  def _step(self, action):
    # action = 0: Hold, 1: Buy, 2: Sell
    rewards = 0
    if action == 0:
      pass
    elif action == 1:
      p =  self.client.get_avg_price(symbol=self.assetpair)* self.position_increment
      if p > self.client.get_asset_balance(asset='USDT')['free']:
        # rewards = -1
        pass
      else:
        order = self.client.order_market_buy(
          symbol=self.assetpair,
          quantity=self.position_increment,
        )
        print("Bought {} of {}".format(self.position_increment, self.asset1))
        self.orders += order['fills']
        rewards += .5 * (self.MACD[-1])
    elif action == 2:
      rewards = 0
      # sell
      quantity = self.client.get_asset_balance(asset='BTC')['free']
      order = self.client.order_market_sell(
        symbol=self.assetpair,
        quantity=quantity,
      )
      print("Sold {} of {}".format(self.position_increment, self.asset1))
      cost_basis = 0
      sell_value = 0
      for o in self.orders:
        cost_basis = o['price']*o['qty']
      for o in order['fills']:
        sell_value = o['price']*o['qty']
        
      self.balance = self.client.get_asset_balance(asset='BTC')['free'] + self.client.get_asset_balance(asset='USDT')['free']
      self.orders = []
      rewards = sell_value - cost_basis

    cur_price = float(self.client.get_avg_price(symbol=self.assetpair)['price'])
    self.return_history.pop(0)
    self.return_history.append(cur_price - self.price_data.iloc[-1,:]['Close'])
    self.price_data = self.price_data.append({'Close': cur_price}, ignore_index=True)
    self.MACD_trend = ta.trend.ema_indicator(self.price_data['Close'], self.fast_ema) - ta.trend.ema_indicator(self.price_data['Close'], self.slow_ema)
    self.MACD_trend = self.MACD_trend.fillna(self.MACD_trend[self.slow_ema]).tolist()
    self.MACD.pop(0)
    self.MACD.append(self.MACD_trend[-1])

    self._state = [self.balance] + self.return_history + self.MACD
    
    return ts.transition(
      np.array(self._state, dtype=np.float32), reward=rewards, discount=0.8)
      
  def action_spec(self):
    return self._action_spec
  
  def _reset(self):
    self._state = [self.balance] + self.return_history + self.MACD
    return ts.restart(np.array(self._state, dtype=np.float32))


  def buy_and_hold(self):
    amount =  self.initial_balance / self.price_data.iloc[0, :]['Close']
    return self.price_data * amount
  