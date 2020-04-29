# Q Network RL Trading bot

This bot uses the the tensorflow DQN algorithm to trade based on price data and current balance. This project is based off of the tensorflow way of modeling agents and environments, adhering to py_environment.PyEnvironment and dqn_agent.DqnAgent base classes.
This project is currently only running past data, and is not running live yet. Price data is received for bitcoin from Binance, but it can be changed.

## Running The bot
```
python trader.py
```

### Get new price data
Generate a new prices dataframe by running prices.py
It will generate a new csv file with info loaded from binance
make sure to set your .env vars for binance
```
CLIENT_KEY=XXXXX
CLIENT_SECRET=XXXXX
python binance-api.py
python stock-api.py
```

### Improvements
Fine tuning hyper parameters
Improve reward function
Saving model and loading model if it exists
Adding sentiment analysis to the state
Live trading 
Risk Management