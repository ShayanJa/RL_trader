import abc
import tensorflow as tf
import numpy as np
import ta

from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils
from tf_agents.specs import array_spec
from tf_agents.environments import wrappers
from tf_agents.trajectories import time_step as ts

tf.compat.v1.enable_v2_behavior()

class BitcoinEnvironment(py_environment.PyEnvironment):
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
    self.position_increment = .1
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
      pass
      # dp = self.price_data.iloc[self.t, :]['Close'] - self.price_data.iloc[self.t-1, :]['Close']
      # for p in self.positions:
      #   rewards += 0.1 * dp
      #   rewards += .2 * (self.MACD[-1])
        # if dp < 0:
        #   rewards *= 2
    elif action == 1:
      p = self.price_data.iloc[self.t, :]['Close'] * self.position_increment
      if p > self.cash_balance:
        # rewards = -1
        pass
      else:
        self.cash_balance -= p 
        self.positions.append(self.price_data.iloc[self.t, :]['Close'])
        # rewards = .5 * self.mean_data.iloc[self.t, :]['Close'] - self.price_data.iloc[self.t, :]['Close'] 
        rewards += .5 * (self.MACD[-1])
    elif action == 2:
      rewards, profits = 0, 0
      for p in self.positions:
        profits += (self.price_data.iloc[self.t, :]['Close'] - p) * self.position_increment
        self.cash_balance += self.price_data.iloc[self.t, :]['Close']* self.position_increment
      self.balance = self.cash_balance
      self.positions = []
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
      np.array(self._state, dtype=np.float32), reward=rewards, discount=0.8)
   
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

  
class LiveBitcoinEnvironment(BitcoinEnvironment):
  def __init__(self, client, price_history_t, mean_history_t, macd_t, fast_ema, slow_ema):
    # get price_data for 1 month ago
    # get client balance
    super(LiveBitcoinEnvironment, self).__init__(10000, price_data, price_history_t, mean_history_t, macd_t, fast_ema, slow_ema)
    self.price_data = client.get_price_data_1mo_1day()
    self.t = len(self.price_data) 
    

  # def run(self, initial_blanace, price_data, price_history_t,mean_history_t, macd_t , fast_ema, slow_ema ):
  #   # Place uptodate price_data
  #   self.t = len(price_data)-1
  #   self.return_history = [self.price_data.iloc[self.t-k, :]['Close'] - self.price_data.iloc[self.t-k-1, :]['Close'] for k in reversed(range(self.price_history_t))]
  #   self.MACD_trend = ta.trend.ema_indicator(self.price_data['Close'], self.fast_ema) - ta.trend.ema_indicator(self.price_data['Close'], self.slow_ema)
  #   self.MACD_trend = self.MACD_trend.fillna(self.MACD_trend[slow_ema]).tolist()
  #   self.MACD = [self.MACD_trend[self.t] for _ in reversed(range(self.macd_t))]
  #   self._state = [self.balance] + self.return_history + self.MACD

  def set_next_price(self, price):
    self.price_data.append(price)

