# Q Network RL Trading bot

This bot uses the the tensorflow DQN algorithm to trade based on price data and current balance. This project is based off of the tensorflow way of modeling agents and environments, adhering to py_environment.PyEnvironment and dqn_agent.DqnAgent base classes.
This project is currently only running past data, and is not running live yet. User discretion is advised if used live.

## Running The bot
```
python trader.py
```

### Get price date
Generate a new prices dataframe by running prices.py
It will generate a new csv file with info loaded from binance
make sure to set your .env vars for binance
```
CLIENT_KEY=XXXXX
CLIENT_SECRET=XXXXX
python prices.py
```

### Future improvements
Fine tuning hyper parameters
Improve reward function
Saving model and loading model if it exists
Adding sentiment analysis to the state
Live trading 
Risk Management