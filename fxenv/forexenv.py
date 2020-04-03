import random
import json
import gym
from collections import deque
from numpy import genfromtxt
from gym import utils
from gym import spaces
import pandas as pd
import numpy as np

MAX_ACCOUNT_BALANCE = 2147483647
MAX_NUM_SHARES = 2147483647
MAX_SHARE_PRICE = 5000
MAX_OPEN_POSITIONS = 5
MAX_STEPS = 20000

INITIAL_ACCOUNT_BALANCE = 10000


class MyForexEnv(gym.Env):

    metadata = {'render.modes': ['human']}


    # Set this in SOME subclasses
    # metadata = {'render.modes': []}
    # reward_range = (-float('inf'), float('inf'))
    # spec = None

    # Set these in ALL subclasses
    # action_space = None
    # observation_space = None

    def __init__(self, dataset='data/myfxchoice_eurusd.CSV'):

        self.capital = 10000
        self.min_stop_loss = 100
        self.min_take_profit = 100
        self.max_stop_loss = 2000
        self.max_take_profit = 1000
        self.leverage = 100

        #the closing cause
        self.c_c = 0
        
        csv_f = dataset 
        self.dataset = dataset
        self.equity = self.capital
        self.balance = self.capital
        self.balance_ant = self.capital

        self.equity_ant = self.capital

        #status nop=0, buy=1, sell=-1
        self.order_status = 0 
        self.order_profit = 0.0

        #select whether to use balance (0) or equity (1) in calculating the reward
        self.bonus_type = 1

        self.order_symbol = 0
        self.reward = 0.0
        
        # Min / Max SL / TP, Min / Max (Default 1000?) in pips 
        self.pip_cost = 0.00001

        #cumulative margin = open_price * volume * 100000 / leverage
        self.margin = 0.0
        # the min order time in the ticks
        self.min_order_time = 1

        #order volume relative to equity
        self.rel_volume = 0.2

        #spread calculus: 0=from last csv column in pips=0
        #lineal from volatility=1, quadratic=2, exponential=3
        self.spread_function = 0
        self.spread = 20
        self.ant_c_c = 0
        self.num_symbols = 1

        #initializ tick counter
        self.tick_count = 0
        
        self.use_return = 0

        self.my_data = genfromtxt(csv_f,delimiter=',')

        self.num_ticks = len(self.my_data)

        self.num_colums = len(self.my_data[0])

        self.preprosessing = 0

        #Normalization method = 0 leaves the data the same, 1 = normalizes, 2 = standardizes,
        #3 = standardizes and truncates range -1.1
        self.norm_method = 1


        def _next_observation(self):
        # Get the stock data points for the last 5 days and scale to between 0-1
        frame = np.array([
            self.df.loc[self.current_step: self.current_step +
                        5, 'Open'].values / MAX_SHARE_PRICE,
            self.df.loc[self.current_step: self.current_step +
                        5, 'High'].values / MAX_SHARE_PRICE,
            self.df.loc[self.current_step: self.current_step +
                        5, 'Low'].values / MAX_SHARE_PRICE,
            self.df.loc[self.current_step: self.current_step +
                        5, 'Close'].values / MAX_SHARE_PRICE,
            self.df.loc[self.current_step: self.current_step +
                        5, 'Volume'].values / MAX_NUM_SHARES,
        ])

        # Append additional data and scale each value to between 0-1
        obs = np.append(frame, [[
            self.balance / MAX_ACCOUNT_BALANCE,
            self.max_net_worth / MAX_ACCOUNT_BALANCE,
            self.shares_held / MAX_NUM_SHARES,
            self.cost_basis / MAX_SHARE_PRICE,
            self.total_shares_sold / MAX_NUM_SHARES,
            self.total_sales_value / (MAX_NUM_SHARES * MAX_SHARE_PRICE),
        ]], axis=0)

        return obs










    def step(self, action):
        pass

    def reset(self):
        pass

    def render(self):
        pass

