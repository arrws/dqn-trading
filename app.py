#!/usr/bin/python
# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import itertools
import sys
import random
import time
import os

from trader import *

import warnings
warnings.filterwarnings("ignore")

# COMMAND LINE INTERFACE
class App:
    def __init__(self, testing=False):
        if not testing:
            self.trader = Trader()

    def main(self):
        os.system('clear')

        # Default Values
        self.trader.agent.restore_network_progress(path="backup/"+n)
        c = "ETH-USD"
        n = "LSTM"
        e = 20
        y, m, d = 2018, 1, 1

        # Main Loop
        while(True):
            print("\n\n TRADER MENU:")
            self.print_params(c=c, n=n, e=e, d=d, m=m, y=y)
            print("\n 1  -  setup params")
            print(" 2  -  run agent on day")
            print(" 3  -  validate agent on day")
            print(" 4  -  start trainning from scratch")
            print(" 5  -  plot month data")
            print(" 6  -  test on synthetic data")
            print(" 0  -  quit\n")

            x = self.get_number(" COMMAND: ")
            if x==1:
                print("\nSETUP PARAMS:")
                print(" 1  -  crypto")
                print(" 2  -  network")
                print(" 3  -  epochs")
                print(" 4  -  day")
                print(" 5  -  month")
                print(" 6  -  year")
                print(" 0  -  back\n")
                x = self.get_number(" COMMAND: ")
                if x==1:
                    c = self.get_crypto()
                elif x==2:
                    new_n = self.get_network()
                    if n != new_n:
                        n = new_n
                        self.trader.agent.restore_network_progress(path="backup/"+n)

                elif x==3:
                    e = self.get_epochs()
                elif x==4:
                    d = self.get_day()
                elif x==5:
                    m = self.get_month()
                elif x==6:
                    y = self.get_year()
                else:
                    print("back...")

                # bounds verification
                if y==2018 and m>4:
                    print("month reset to present")
                    m=4
                if (c=='ETH-USD' or c=='LTC-USD') and y<2017:
                    print("year reset to firs viable")
                    y=2017

            elif x==2:
                self.print_params(c=c, n=n, d=d, m=m, y=y)
                self.trader.train_on_day(crypto=c, day=d, month=m, year=y)

            elif x==3:
                self.print_params(c=c, n=n, d=d, m=m, y=y)
                self.trader.validate_on_day(crypto=c, day=d, month=m, year=y)

            elif x==4:
                self.print_params(c=c, n=n, e=e, m=m, y=y)
                self.trader.train_from_scratch(crypto=c, epochs=e, month=m, year=y)
            elif x==5:
                self.print_params(c=c, m=m, y=y)
                self.trader.overview_month(c, m ,y)
            elif x==6:
                self.print_params(e=e)
                self.trader.test_on_synthetic_data(e)
            elif x==0:
                print("quitting ...\n")
                return;
            else:
                print("wrong command\n")
            #os.system('clear')


    def print_params(self, c=None, n=None, e=None, d=None, m=None, y=None):
        s = '\n PARAMS: '
        if c:
            s += " Crypto: "+c+"  "
        if n:
            s += " Network: "+n+"  "
        if e:
            s += " Epochs: "+str(e)+"  "
        if d:
            s += " Day: "+str(d)+"  "
        if m:
            s += " Month: "+str(m)+"  "
        if y:
            s += " Year: "+str(y)+"  "
        print(s)


    def get_year(self):
        x = self.get_number("\nyear (2016-2018): ")
        while x not in range(2016, 2019):
            print("bad input")
            x = self.get_number("year: ")
        return x

    def get_month(self):
        x = self.get_number("\nmonth (1-12): ")
        while x not in range(1, 13):
            print("bad input")
            x = self.get_number("month: ")
        return x

    def get_day(self):
        x = self.get_number("\nday (1-27): ")
        while x not in range(1, 28):
            print("bad input")
            x = self.get_number("day: ")
        return x

    def get_epochs(self):
        x = self.get_number("\nepochs (5-100): ")
        while x not in range(5, 101):
            print("bad input")
            x = self.get_number("epochs: ")
        return x

    def get_crypto(self):
        ids = ['BTC-USD', 'ETH-USD', 'LTC-USD']
        x = self.get_number("\n1 - Bitcoin\n2 - Ethereum\n3 - Litecoin\nchoose currency: ")
        while x not in range(1, 4):
            print("bad input")
            x = self.get_number("\nchoose currency: ")
        return ids[x-1]

    def get_network(self):
        ids = ['SIMPLE', 'LSTM', 'DLSTM']
        x = self.get_number("\n1 - Simple Network\n2 - LSTM Network\n3 - Deep LSTM Network\nchoose network: ")
        while x not in range(1, 4):
            print("bad input")
            x = self.get_number("\nchoose network: ")
        return ids[x-1]

    def get_number(self, s):
        while True:
            try:
                num = abs(int(input(s)))
                return num
            except ValueError:
                print("bad input")


t = App()
t.main()


