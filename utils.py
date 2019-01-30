#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import random
import math
import os

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.finance import candlestick_ohlc
# NOT DEPRECATED
#from mpl_finance import candlestick_ohlc



# LOGGING SETUP

PATH = "output"
try:
    os.makedirs(PATH+"/")
    os.makedirs(PATH+"/nn/")
except OSError:
    pass
logfile = open(PATH+"/log.txt","a")


# PRINTING FUNCTIONS

def print_daily_score(day, p, s, d, e):
    log_print('day: {:d}\t  profit: {:.2f}\t  sharpe: {:.2f}\t  drawdawn: {:.2f}\t  exvar: {:.2f}'.format(day, p, s, d, e))

def print_final_score(p, s, d, e):
    log_print("PROFIT {:.2F}\t  SHARPE {:.2F}\t  DRAWDOWN {:.2F}\t  EXVAR {:.2F}".format(p, s, d, e))

def log_print(s):
    print(s)
    logfile.write(s+'\n')
    logfile.flush()


# MISC FUNCTIONS

def vectorize(n, x):
    v = np.zeros([n])
    v[x] = 1
    return v

def convert_nan(v):
    f = []
    for x in v:
        if math.isnan(x) or math.isinf(x):
            f.append(-1)
        else:
            f.append(x)
    return f

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:]-ret[:-n]
    return ret[n-1:]/n


# PLOTTING FUNCTIONS

def plot_candlestick(name, data, actions=None):
    # plot candlestick chart and optionally the associated trade signal

    plt.clf()

    idx = [k for k in range(len(data))]
    quotes = [tuple([idx[i], data[i][0], data[i][1], data[i][2], data[i][3]]) for i in range(len(data))]

    ax1 = plt.subplot2grid((5, 4), (0, 0), rowspan=4, colspan=4)
    candlestick_ohlc(ax1, quotes, width=0.6, colorup='#77d879', colordown='#db3f3f')
    ax1.grid(True)
    plt.ylabel('Stock Price')

    ax2 = plt.subplot2grid((5, 4), (4, 0), sharex=ax1, rowspan=1, colspan=4)
    ax2.bar(idx, [data[i][4] for i in range(len(data))], color='#8CBED6')
    ax2.grid(True)
    plt.ylabel('Volume')

    if actions != None:
        offset = 5
        for i in range(len(actions)-1):
            color = "black"
            if actions[i] == 2:
                color = "red"
            elif actions[i] == 1:
                color = "green"
            ax1.plot([i, i+1], [data[i][2]-offset, data[i+1][2]-offset], color)

    plt.xlabel('Time')
    plt.subplots_adjust(left=0.06, bottom=.12, right=.99, wspace=.10, hspace=.10)

    figure = plt.gcf() # get current figure
    figure.set_size_inches(12, 5)

    s = PATH+"/"+name+"_candlestick"
    print("...generated: "+s)
    plt.savefig(s, dpi=160)


def plot_final_eval(label, name, v, base_v=None, base_v2=None):
    # plots the scores of different trading days
    plt.clf()

    idx = [i+1 for i in range(len(v))]
    plt.plot(idx, v, color='royalblue', label='agent')

    if base_v:
        plt.plot(idx,[base_v for _ in idx], color='red', label='momentum')

    plt.legend()
    plt.ylabel(label)
    plt.xlabel('Epochs')

    s = PATH+"/final_"+label+"_"+name
    print("...generated: "+s)
    plt.savefig(s, dpi=160)


def plot_epoch_profit(name, profits, base_profits, base2_profits):
    # plots the trainning and baselines profit of different trading days
    plt.clf()

    idx = [i+1 for i in range(len(profits))]
    plt.plot(idx, profits, color='royalblue', label='agent')

    idx = [i+1 for i in range(len(base_profits))]
    plt.plot(idx,[x for x in base_profits], color='red', label='momentum')

    idx = [i+1 for i in range(len(base2_profits))]
    plt.plot(idx,[x for x in base2_profits], color='tomato', label='buynhold')

    plt.legend()
    plt.ylabel('Profits')
    plt.xlabel('Days')

    s = PATH+"/profits_"+name
    print("...generated: "+s)
    plt.savefig(s, dpi=160)


def plot_epoch_diff(name, profits, base_profits):
    # plots the trainning profit compared to the baselines of different trading days
    plt.clf()

    idx = [i for i in range(len(profits))]
    diff = []
    colors = []
    for i in range(len(profits)):
        diff.append(profits[i] - base_profits[i])
        if diff[-1] < 0:
            colors.append("red")
        else:
            colors.append("green")
    #colors = itertools.cycle(colors)
    plt.bar(idx, diff, color = colors)

    plt.legend()
    plt.ylabel('Profits')
    plt.xlabel('Days')

    s = PATH+"/diff_"+name
    print("...generated: "+s)
    plt.savefig(s, dpi=160)


def plot_cumulative_gains(name, test_profits=None, base_profits=None, base_profits2=None):
    # plots the accumulated profit of different trading days
    plt.clf()
    idx = [k for k in range(len(test_profits))]

    if test_profits:
        plt.plot(idx, test_profits, color='royalblue', label='agent')

    if base_profits:
        plt.plot(idx,[x for x in base_profits], color='red', label='momentum')

    if base_profits2:
        plt.plot(idx,[x for x in base_profits2], color='tomato', label='buynhold')

    plt.legend()
    plt.ylabel('Profits')
    plt.xlabel('Time')

    s = PATH+"/gains_"+name
    print("...generated: "+s)
    plt.savefig(s, dpi=160)


def plot_bollinger_bands(name, data):
    # plots the bollinger bands
    plt.clf()
    window = 10

    idx = [k for k in range(len(data))]
    plt.plot(idx,[x[3] for x in data])

    close =  np.array(data)[:,3]
    rm = pd.rolling_mean(close, window=window)
    rstd = pd.rolling_std(close, window=window)

    # NOT DEPRECATED
    #close =  pd.Series(np.array(data)[:,3])
    #r = close.rolling(window)
    #rm = r.mean().values
    #rstd = r.std().values

    for i in range(window):
        rm[i] = rm[window]
        rstd[i] = rstd[window]

    up = rm + 2*rstd
    low = rm - 2*rstd

    plt.plot(idx,rm, color='black', label='rolling mean')
    plt.plot(idx,up, color='green', label='upper band')
    plt.plot(idx,low, color='red', label='lower band')

    plt.legend()
    plt.ylabel('Price')
    plt.xlabel('Time')

    s = PATH+"/"+name+"_bollinger_bands"
    print("...generated: "+s)
    plt.savefig(s, dpi=160)


def plot_returns_variation(name, data):
    # plots the return variation
    plt.clf()
    idx = [k for k in range(len(data))]
    plt.plot(idx,[x[3]-x[0] for x in data])
    plt.ylabel('Daily Return')
    plt.xlabel('Time')

    s = PATH+"/"+name+"_returns_variation"
    print("...generated: "+s)
    plt.savefig(s, dpi=160)


