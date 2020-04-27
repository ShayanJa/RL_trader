import abc
import tensorflow as tf
import numpy as np

from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils
from tf_agents.specs import array_spec
from tf_agents.environments import wrappers
from tf_agents.trajectories import time_step as ts


tf.compat.v1.enable_v2_behavior()

class BitcoinEnvironment(py_environment.PyEnvironment):
  def __init__(self, initial_balance, price_data, price_history_t):
    self.t = 0
    self.price_data = price_data
    self.price_history_t = price_history_t
    self.initial_balance = initial_balance
    self.balance = initial_balance
    self.cash_balance = initial_balance
    self.position_increment = .1
    self.positions = []
    self._action_spec = array_spec.BoundedArraySpec(
      shape=(), dtype=np.int32, minimum=0, maximum=2, name='action')
    self._observation_spec = array_spec.BoundedArraySpec(
      shape=(price_history_t+1,), dtype=np.float32, name="observation")
    self._state = [self.balance] + [self.price_data.iloc[self.t - k, :]['Close'] for k in reversed(range(price_history_t))]
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
    elif action == 1:
      p = self.price_data.iloc[self.t, :]['Close'] * self.position_increment
      if p > self.cash_balance:
        rewards = -1
      else:
        self.cash_balance -= p 
        self.positions.append(self.price_data.iloc[self.t, :]['Close'])
    elif action == 2:
      rewards, profits = 0, 0
      for p in self.positions:
        profits += (self.price_data.iloc[self.t, :]['Close'] - p) * self.position_increment
        self.cash_balance += self.price_data.iloc[self.t, :]['Close']* self.position_increment
      self.balance = self.cash_balance
      self.positions = []
      if profits > 0:
        rewards = 2 * profits
      else:
        rewards = profits
        
    self.balance = self.cash_balance
    for p in self.positions:
      self.balance += self.price_data.iloc[self.t, :]['Close'] * self.position_increment
    self.t += 1
    self.price_history.pop(0)
    self.price_history.append(self.price_data.iloc[self.t, :]['Close'])

    if self.t == len(self.price_data -1):
      self._episode_ended = True

    self._state = [self.balance] + self.price_history
    
    return ts.transition(
      np.array(self._state, dtype=np.float32), reward=rewards, discount=0.9)
   
  def action_spec(self):
    return self._action_spec
  
  def _reset(self):
    self.t = 0
    self._episode_ended = False
    self.profits = 0
    self.balance = self.initial_balance
    self.cash_balance = self.initial_balance
    self.positions = []
    self.price_history = [0 for i in range(self.price_history_t)]
    self._state = [self.balance] + self.price_history
    return ts.restart(np.array(self._state, dtype=np.float32))