#!/usr/bin/python
# -*- coding: utf-8 -*-
import time
import numpy as np
import requests
from datetime import datetime, timedelta
import pandas
import warnings

class Broker:

    def __init__(self, product_id='ETH-USD', verbose=False):
        warnings.filterwarnings("ignore")
        self.url = 'https://api.gdax.com'.rstrip('/')
        self.verbose = verbose
        self.test = False
        self.data = []

        # granularity: 60, 300, 900, 3600, 21600, 86400
        #              one minute, five minutes, fifteen minutes, one hour, six hours, and one day
        self.granularity = 300
        self.product_id = product_id
        self.synthetic_data = False


    def set_crypto(self, crypto):
        # change set cryptocurrency
        self.product_id = crypto


    def set_date(self, day=1, month=7, year=2017):
        # change set date
        if not self.synthetic_data:
            self.make_order(day, month, year)
        self.k = 0


    def make_order(self, day=1, month=7, year=2017):
        # makes an order for a date
        if self.verbose:
            print ('ORDER DAY: ', day)
        self.get_prices(datetime(year, month, day), datetime(year, month, day + 1))


    def get_prices(self, start=None, end=None):
        # sends a request for data
        params = {}
        if start is not None:
            params['start'] = start
        if end is not None:
            params['end'] = end
        params['granularity'] = self.granularity

        path = self.url + '/products/{}/candles'.format(self.product_id)

        if self.verbose:
            print (self.product_id, 'request for:', start, end)

        # [ time, low, high, open, close, volume ]
        time.sleep(.3)
        r = requests.get(path, params=params, timeout=30)
        #print(r.json())

        self.data = pandas.DataFrame(data=r.json(), columns=['time', 'low', 'high', 'open', 'close', 'volume'])
        self.data = self.data.reindex_axis(['time', 'open', 'high', 'low', 'close', 'volume'], axis=1)
        self.data.set_index('time', inplace=True)
        self.data = self.data.iloc[::-1]

        if self.verbose:
            print (self.product_id, 'received:', len(self.data))


    def generate_synthetic_data(self):
        # generates a sin-like trading price series
        self.synthetic_data = True
        ln = 200
        x = np.sin(np.arange(ln)/10.0)
        y = np.sin(np.arange(ln)/5.0)
        z = np.sin(np.arange(ln)/3.0)
        x = np.add(np.add(x,y),z)
        self.data = [[x[i]+10, max(x[i],x[i+1])+0.1+10, min(x[i],x[i+1])-0.1+10, x[i+1]+10, 10*max(x[i],x[i+1])] for i in range(len(x)-1)]


    def check_last_trade(self):
        # cheks for the last trade
        if self.k == len(self.data)-1:
            return True
        return False


    def get_data(self):
        # for testing purposes
        if self.synthetic_data:
            x = self.data[self.k]
            self.k += 1
            if self.k+1 == len(self.data):
                self.synthetic_data = False
                return [True, x]
            return [False, x]

        # returns [ open, high, low, close, volume ] formatted data
        done = self.check_last_trade()
        x = self.data.iloc[self.k].values.tolist()
        self.k += 1
        return [done, x]


