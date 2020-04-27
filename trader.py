import tensorflow as tf

from tf_agents.agents.dqn import dqn_agent
from tf_agents.networks import q_network
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import tf_py_environment
from tf_agents.trajectories import trajectory
from tf_agents.environments import wrappers
from tf_agents.metrics import tf_metrics
from tf_agents.policies import random_tf_policy
from tf_agents.policies import policy_saver
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.utils import common
from tf_agents.metrics import py_metrics
from tf_agents.metrics import tf_metrics
from tf_agents.drivers import py_driver
from tf_agents.drivers import dynamic_episode_driver

from environment import BitcoinEnvironment
import datetime as dt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def compute_performance(environment, policy):

    time_step = environment.reset()
    total_return = 0.0
    balance = [time_step.observation[0][0]]

    while not time_step.is_last():
      action_step = policy.action(time_step)
      time_step = environment.step(action_step.action)
      total_return += time_step.reward
      balance.append(time_step.observation[0][0])
      print(time_step.observation)

    return total_return[0], balance

num_iterations = 10000  # @param

replay_buffer_capacity = 100000  # @param

fc_layer_params = (100,23,33)

batch_size = 100  # @param
learning_rate = 1e-4  # @param
log_interval = 200  # @param

num_eval_episodes = 2  # @param
eval_interval = 1000  # @param

date_split = dt.datetime(2018, 3, 16, 1, 0)  # @param

# Split data into training and test set
prices = pd.read_csv('btcusd.csv', parse_dates=True, index_col=0)
train = prices[:date_split]
test = prices[date_split:]

initial_balance = 10000  # @param
training_duration = 200  # @param
eval_duration = 200 # @param


# Create Environments
train_py_env = wrappers.TimeLimit(BitcoinEnvironment(initial_balance, train, 15, 15, 10, 12, 26), duration=training_duration)
eval_py_env = wrappers.TimeLimit(BitcoinEnvironment(initial_balance, test, 15, 15, 10, 12, 26 ), duration=eval_duration)
test_py_env = wrappers.TimeLimit(BitcoinEnvironment(initial_balance, test, 15, 15, 10, 12, 26 ), duration=len(test)-1)

train_env = tf_py_environment.TFPyEnvironment(train_py_env)
eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)
test_env = tf_py_environment.TFPyEnvironment(test_py_env)

# Initialize Q Network
q_net = q_network.QNetwork(
  train_env.observation_spec(),
  train_env.action_spec(),
  fc_layer_params=fc_layer_params)

optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)

train_step_counter = tf.compat.v2.Variable(0)

tf_agent = dqn_agent.DqnAgent(
  train_env.time_step_spec(),
  train_env.action_spec(),
  q_network=q_net,
  optimizer=optimizer,
  train_step_counter=train_step_counter)

tf_agent.initialize()

eval_policy = tf_agent.policy
collect_policy = tf_agent.collect_policy

replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
  data_spec=tf_agent.collect_data_spec,
  batch_size=train_env.batch_size,
  max_length=replay_buffer_capacity)

replay_observer = [replay_buffer.add_batch]

train_metrics = [
  tf_metrics.NumberOfEpisodes(),
  tf_metrics.EnvironmentSteps(),
  tf_metrics.AverageReturnMetric(),
  tf_metrics.AverageEpisodeLengthMetric(),
]

def collect_step(environment, policy):
  time_step = environment.current_time_step()
  action_step = policy.action(time_step)
  next_time_step = environment.step(action_step.action)
  traj = trajectory.from_transition(time_step, action_step, next_time_step)

  # Add trajectory to the replay buffer
  replay_buffer.add_batch(traj)

for _ in range(1000):
  collect_step(train_env, tf_agent.collect_policy)

dataset = replay_buffer.as_dataset(
  num_parallel_calls=3,
  sample_batch_size=batch_size,
  num_steps=2).prefetch(3)

driver = dynamic_step_driver.DynamicStepDriver(
  train_env,
  collect_policy,
  observers=replay_observer + train_metrics,
  num_steps=1)

iterator = iter(dataset)

print("Initial Balance: {}".format(initial_balance))

tf_agent.train = common.function(tf_agent.train)
tf_agent.train_step_counter.assign(0)

final_time_step, policy_state = driver.run()

for i in range(1000):
  final_time_step, _ = driver.run(final_time_step, policy_state)

episode_len = []
portfolio_balance = []
for i in range(num_iterations):
  final_time_step, _ = driver.run(final_time_step, policy_state)
  for _ in range(1):
     collect_step(train_env, tf_agent.collect_policy)

  experience, _ = next(iterator)
  train_loss = tf_agent.train(experience=experience)
  step = tf_agent.train_step_counter.numpy()

  if step % log_interval == 0:
    print('step = {0}: loss = {1}'.format(step, train_loss.loss))
    episode_len.append(train_metrics[3].result().numpy())
    print('Average episode length: {}'.format(train_metrics[3].result().numpy()))

  if step % eval_interval == 0:
    reward, portfolio_balance = compute_performance(eval_env, tf_agent.policy)
    print('step = {0}: Average Reward = {1}: Ending Portfolio Balance = {2}'.format(step, reward, portfolio_balance[-1]))

# my_policy = tf_agent.collect_policy
# saver = policy_saver.PolicySaver(my_policy, batch_size=None)
# saver.save('policy_trader')

# Compare against Buy and hold
reward, portfolio_balance = compute_performance(test_env, tf_agent.policy)
bnh = eval_py_env.buy_and_hold().head(eval_duration+1)
portfolio_balance = pd.DataFrame(data={'Close': np.array(portfolio_balance)}, index=bnh.index)
print("RL GAIN = {}: BUY/HOLD = {}".format(portfolio_balance.iloc[-1,:]['Close']-portfolio_balance.iloc[0,:]['Close'], bnh.iloc[-1]['Close']-bnh.iloc[0]['Close']))
bnh['Close'].plot()
portfolio_balance['Close'].plot()

history = pd.DataFrame(data={"Buy/Hold": bnh['Close'], "RL": portfolio_balance['Close']}, index=bnh.index)
history.to_csv("results/trader-{}".format(dt.datetime.now()))
plt.show()