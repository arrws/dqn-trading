#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import random
import math
import warnings

from scipy import stats

import pandas as pd

from talib.abstract import *
from utils import *


class Quant:
    def __init__(self):
        warnings.filterwarnings("ignore")
        self.reinit()
        self.unit = 100 # units of Bitcoin traded each time
        self.k = 0
        self.reinit()

    def reinit(self):
        self.data = [] # [ open, high, low, close, volume ]
        self.features = [] # [ financial_indices ]
        self.actions = [] # [ 0/1/2 ]
        self.returns = [] # [ trade_return ]
        self.dones = [] # [ True/False ]

        #self.rsharpes = []
        self.sharpes = []


    def add_transition(self, s, a, r, d):
        # adds trade details to memmory
        self.data.append(s)
        self.features.append(self.compute_feature())
        self.actions.append(a)
        self.returns.append(r)
        self.dones.append(d)


    def sample_batches(self, no_actions, batch_size):
        # samples random transitions from the replay memory
        if batch_size > len(self.data):
            indexes = [x for x in range(len(self.data))]
        else:
            indexes = random.sample(range(len(self.data)), batch_size-1)

        s_batch = np.array([self.features[i] for i in indexes])
        a_batch = np.array([vectorize(no_actions, self.actions[i]) for i in indexes])
        r_batch = np.array([self.returns[i] for i in indexes])
        #r_batch = np.array([self.rsharpes[i] for i in indexes])
        s2_batch = np.array([self.features[i-1] for i in indexes])
        d_batch = np.array([self.dones[i] for i in indexes])

        return [s_batch, a_batch, r_batch, s2_batch, d_batch]


    def compute_feature(self):
        # computes financial indicators for the last added data
        data = np.array(self.data)
        inputs = {
            'open': data[:, 0],
            'high': data[:, 1],
            'low': data[:, 2],
            'close': data[:, 3],
            'volume': data[:, 4],
            }

        diff = np.diff(inputs['close'])
        diff = np.insert(diff, 0, 0)

        sma15 = SMA(inputs, timeperiod=10, price='close')
        sma60 = SMA(inputs, timeperiod=20, price='close')

        rsi = RSI(inputs, timeperiod=10, price='close')
        atr = ATR(inputs, timeperiod=10, prices=['high', 'low', 'open'])

        z_price = stats.zscore(inputs['close'])
        z_volume = stats.zscore(inputs['volume'])

        feature = [
                diff[-1],
                inputs['close'][-1] - inputs['open'][-1],
                z_price[-1]/10,
                z_volume[-1],
                rsi[-1]/100,
                atr[-1],
                inputs['close'][-1] - sma15[-1],
                sma15[-1] - sma60[-1],
                ]

        feature = convert_nan(feature)
        return feature


    def get_signal_history(self, limit=999):
        return self.actions[-limit:]

    def get_data_history(self, limit=999):
        return self.data[-limit:]

    def get_last_action(self):
        return [self.actions[-1]]

    def get_last_features(self):
        return np.array([self.features[-1]])

    def get_total_return(self):
        return sum(self.returns)

    def get_cumulative_returns(self):
        r = [0]
        for x in self.returns:
            r.append(r[-1]+x)
        return r


    def get_return(self, a, s):
        # computes the profit made
        # data format: [ open, high, low, close, volume ]
        # action format: 0-hold, 1-buy, 2-sell
        x = 0
        if a == 1:
            x = 1
        elif a == 2:
            x = -1

        r = x * (s[3] - self.data[-1][3]) * self.unit

        # add transaction fee
        if self.actions[-1] != a:
            r -= abs(s[0]*0.01)
        return r


    def get_last_trend(self):
        # for momentum baseline agent
        diff = 0
        if len(self.data)>2:
            #diff = self.data[-1][3] - self.data[-2][3]
            diff = self.data[-1][3] - self.data[-2][3]
        a = 0
        if diff > 0:
            a = 1
        elif diff < 0:
            a = 2
        return a


    def get_last_bollinger(self):
        # for buynhold baseline agent
        window = 20
        data = np.array(self.data[-window:])

        close =  data[:, 3]
        rm = pd.rolling_mean(close, window=window) # DEPRECATED
        rstd = pd.rolling_std(close, window=window)
        up = rm[-1] + 2*rstd[-1]
        low = rm[-1] - 2*rstd[-1]

        # NOT DEPRECATED
        #close =  pd.Series(data[:, 3])
        #r = close.rolling(window)
        #rm = r.mean()
        #rstd = r.std()
        #up = rm.index[-1] + 2*rstd.index[-1]
        #low = rm.index[-1] - 2*rstd.index[-1]

        a = 0
        if data[-1][3] > up:
            a = 2
        elif data[-1][3] < low:
            a = 1
        return a



    # RISK METRICS FUNCTIONS

    def eval_performance(self):
        return [self.get_total_return(), self.get_sharpe_ratio(), self.get_max_drawdown(), self.get_excess_var()]

    def get_returns_vol(self):
        # Return the standard deviation of returns
        return np.std(self.returns)

    def get_returns_mean(self):
        # Return the mean of returns
        return np.mean(self.returns)

    def get_sharpe_ratio(self, rf=0.06):
        # Return the Sharpe ration
        return (self.get_returns_mean() - rf) / self.get_returns_vol()

    def get_drawdown(self):
        maxs = np.maximum.accumulate(self.returns)
        dds = 1 - self.returns / maxs
        return dds

    def get_max_drawdown(self):
        drawdowns = self.get_drawdown()
        maxdd = np.max(convert_nan(drawdowns))
        return maxdd

    def get_var(self, alpha):
        # Returns the excess
        sorted_returns = np.sort(self.returns)
        index = int(alpha * len(sorted_returns))
        return abs(sorted_returns[index])

    def get_excess_var(self, rf=0.06, alpha=0.05):
        # Returns the excess Value at Risk for all return series
        return (self.get_returns_mean() - rf) / self.get_var(alpha)



