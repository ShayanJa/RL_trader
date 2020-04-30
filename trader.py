import tensorflow as tf

from tf_agents.agents.dqn import dqn_agent
from tf_agents.environments import wrappers
from tf_agents.environments import tf_py_environment
from environment import LiveBinanceEnvironment
import pandas as pd
from tf_agents.networks import q_network
import pickle
import os 
from dotenv import load_dotenv
from binance.client import Client
import sched, time

load_dotenv()

AGENT_MODEL_PATH = "policy_10000"

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if physical_devices:
  tf.config.experimental.set_memory_growth(physical_devices[0], True)


# Initialize Binance client
api_key = os.getenv("CLIENT_KEY")
api_secret = os.getenv("SECRET_KEY")
client = Client(api_key, api_secret)

live_env = LiveBinanceEnvironment("BTC", "USDT", .002, 15, 15, 10, 12, 26)
live_env = tf_py_environment.TFPyEnvironment(live_env )
fc_layer_params = (100,23,33)

try:
  collect_policy = tf.compat.v2.saved_model.load(AGENT_MODEL_PATH)
  policy_state = collect_policy.get_initial_state(batch_size=3)
  print("Policy loaded from: {}".format(AGENT_MODEL_PATH))
except:
  raise Exception("Model needed")


policy = tf.compat.v2.saved_model.load(AGENT_MODEL_PATH)
policy_state = collect_policy.get_initial_state(batch_size=3)

# Run a step every x period
s = sched.scheduler(time.time, time.sleep)

time_step = live_env.current_time_step()
def run_step(step, sc):
  action_step = policy.action(step)
  next_time_step = live_env.step(action_step.action)
  time_step = next_time_step
  print(action_step.action)
  print(time_step.observation)
  print(time_step.reward)

  s.enter(60, 1, run_step, (step, sc,))

s.enter(60, 1, run_step, (time_step, s, ))
s.run()
