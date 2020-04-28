import tensorflow as tf
from tf_agents.environments import wrappers
from tf_agents.environments import tf_py_environment
from environment import LiveBitcoinEnvironment, get_binance_price_data
import pandas as pd
from tf_agents.networks import q_network
import pickle
import os 
from binance import client

AGENT_MODEL_PATH = "policy_10000"

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if physical_devices:
  tf.config.experimental.set_memory_growth(physical_devices[0], True)

prices = pd.read_csv('btcusd.csv', parse_dates=True, index_col=0)

live_env = LiveBitcoinEnvironment(client, 15, 15, 10, 12, 26)
fc_layer_params = (100,23,33)

try:
  collect_policy = tf.compat.v2.saved_model.load(AGENT_MODEL_PATH)
  policy_state = collect_policy.get_initial_state(batch_size=3)
except:
  raise Exception("Model needed")

# # Initialize Q Network
# q_net = q_network.QNetwork(
#   live_env.observation_spec(),
#   live_env.action_spec(),
#   fc_layer_params=fc_layer_params)

# optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
# train_step_counter = tf.compat.v2.Variable(0)    

# tf_agent = dqn_agent.DqnAgent(
# train_env.time_step_spec(),
# train_env.action_spec(),
#   q_network=q_net,
#   optimizer=optimizer,
#   train_step_counter=train_step_counter)

# tf_agent.initialize()

def compute_performance(environment, policy):

  time_step = environment.reset()
  total_return = 0.0
  balance = [time_step.observation[0][0]]

  while not time_step.is_last():
    action_step = policy.action(time_step, policy_state)
    time_step = environment.step(action_step.action)
    total_return += time_step.reward
    balance.append(time_step.observation[0][0])
    # print(time_step.observation)

  return total_return[0], balance


# def schedule( fn, time, frequency):
#   fn()