# Q Network RL Trading bot

This bot uses the DQN algorithm to learn a trading strategy based on relevant market data such as price, price history, and technicals. The trading strategy tries to optimize a policy π which maps current knowledge (or s) to the best action a = π(s). This is determined by the reward that is received by this assignment Q(s,a). The bot will try to maximize the reward as much as possible by modifying the policy.
 
The challenge to making a successful RL trading bot is to construct an appropriate reward function and an appropriate state to learn from. 
What factors are more important and what factors matter less?

This project is based off of the tf_agents library, adhering to py_environment.PyEnvironment and dqn_agent.DqnAgent base classes.

For more info on Q-Learning
https://en.wikipedia.org/wiki/Q-learning

## Install
```
pip install -r requirements.txt
```

## Learn 
modify hyper parameters in learner.py
then run
```
python learner.py
```

## Live trading 
```
export CLIENT_KEY=XXXXX
export CLIENT_SECRET=XXXXX
python trader.py
```

### Get new price data
Generate a new prices csv file
It will generate a new csv file with info loaded from binance
make sure to set your .env vars for binance
```
CLIENT_KEY=XXXXX
CLIENT_SECRET=XXXXX
python scripts/binance-prices.py price BTCUSDT
python scripts/stock-prices.py price 
```

### Improvements
- Fine tuning hyper parameters
- Improve reward function
- Adding sentiment analysis to the state
- Risk Management
