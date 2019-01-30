#!/usr/bin/python
# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import itertools
import sys
import random
import time
import os

from agent import *
from utils import *

#start_time = time.time()
#current_time = time.time()

class Trader:
    def __init__(self):
        self.max_days = 25
        self.test_day = self.max_days+1
        self.network_type = "SIMPLE"

        self.agent = Agent(self.network_type)


    def train_from_scratch(self, crypto='ETH-USD', epochs=10, month=7, year=2017):
        self.agent.reinit(crypto=crypto, epochs=epochs, max_days=self.max_days)

        base2_scores = self.run_baseline("BUYNHOLD", month, year)
        base2_profits = np.array(base2_scores)[:,0]
        base2_profit, base2_sharpe, base2_drawdown, base2_exvar = self.unpack_scores(base2_scores)
        print_final_score(base2_profit, base2_sharpe, base2_drawdown, base2_exvar)

        log_print("\nvalidating ...")
        score = self.agent.start_baseline_trading("BUYNHOLD", self.test_day, month, year)
        print_daily_score(self.test_day, score[0], score[1], score[2], score[3])
        test_base2_cum1 = self.agent.quant.get_cumulative_returns()
        score = self.agent.start_baseline_trading("BUYNHOLD", self.test_day+1, month, year)
        print_daily_score(self.test_day, score[0], score[1], score[2], score[3])
        test_base2_cum2 = self.agent.quant.get_cumulative_returns()
        score = self.agent.start_baseline_trading("BUYNHOLD", self.test_day+2, month, year)
        print_daily_score(self.test_day, score[0], score[1], score[2], score[3])
        test_base2_cum3 = self.agent.quant.get_cumulative_returns()


        base_scores = self.run_baseline("MOMENTUM", month, year)
        base_profits = np.array(base_scores)[:,0]
        base_profit, base_sharpe, base_drawdown, base_exvar = self.unpack_scores(base_scores)
        print_final_score(base_profit, base_sharpe, base_drawdown, base_exvar)

        log_print("\nvalidating ...")
        score = self.agent.start_baseline_trading("MOMENTUM", self.test_day, month, year)
        print_daily_score(self.test_day, score[0], score[1], score[2], score[3])
        test_base_cum1 = self.agent.quant.get_cumulative_returns()
        score = self.agent.start_baseline_trading("MOMENTUM", self.test_day+1, month, year)
        print_daily_score(self.test_day, score[0], score[1], score[2], score[3])
        test_base_cum2 = self.agent.quant.get_cumulative_returns()
        score = self.agent.start_baseline_trading("MOMENTUM", self.test_day+2, month, year)
        print_daily_score(self.test_day, score[0], score[1], score[2], score[3])
        test_base_cum3 = self.agent.quant.get_cumulative_returns()

        profits = []
        sharpes = []
        drawdowns = []
        exvars = []

        log_print("\nstarting trainning on month ...")
        for epoch in range(1, epochs+1):
            log_print("\nEPOCH "+str(epoch)+" trainning ...")
            train_scores = []

            for day in range(1, self.max_days+1):
                #trainning
                score = self.agent.start_train_trading(day, month, year)
                train_scores.append(score)
                print_daily_score(day, score[0], score[1], score[2], score[3])

            self.agent.save_network_progress(str(epoch))

            p, s, d, e = self.unpack_scores(train_scores)
            profits.append(p)
            sharpes.append(s)
            drawdowns.append(d)
            exvars.append(e)
            print_final_score(profits[-1], sharpes[-1], drawdowns[-1], exvars[-1])

            log_print("\nvalidating ...")
            score = self.agent.start_trading(self.test_day, month, year)
            print_daily_score(self.test_day, score[0], score[1], score[2], score[3])
            test_cum1 = self.agent.quant.get_cumulative_returns()
            score = self.agent.start_trading(self.test_day+1, month, year)
            print_daily_score(self.test_day, score[0], score[1], score[2], score[3])
            test_cum2 = self.agent.quant.get_cumulative_returns()
            score = self.agent.start_trading(self.test_day+2, month, year)
            print_daily_score(self.test_day, score[0], score[1], score[2], score[3])
            test_cum3 = self.agent.quant.get_cumulative_returns()

            plot_cumulative_gains("1_epoch_"+str(epoch), test_cum1, test_base_cum1, test_base2_cum1)
            plot_cumulative_gains("2_epoch_"+str(epoch), test_cum2, test_base_cum2, test_base2_cum2)
            plot_cumulative_gains("3_epoch_"+str(epoch), test_cum3, test_base_cum3, test_base2_cum3)

            train_profits = np.array(train_scores)[:,0]
            plot_epoch_profit("epoch_"+str(epoch), train_profits, base_profits, base2_profits)
            plot_epoch_diff("epoch_"+str(epoch), train_profits, base_profits)

            if epoch%5 == 0:
                print("")
                plot_final_eval("Profit", str(epoch), profits, base_profit, base2_profit)
                plot_final_eval("Sharpe", str(epoch), sharpes, base_sharpe, base2_sharpe)
                plot_final_eval("Drawdown", str(epoch), drawdowns, base_drawdown, base2_drawdown)
                plot_final_eval("Exvar", str(epoch), exvars, base_exvar, base2_exvar)

        print("")
        plot_final_eval("Profit", "end", profits, base_profit, base2_profit)
        plot_final_eval("Sharpe", "end", sharpes, base_sharpe, base2_sharpe)
        plot_final_eval("Drawdown", "end", drawdowns, base_drawdown, base2_drawdown)
        plot_final_eval("Exvar", "end", exvars, base_exvar, base2_exvar)



    def train_on_day(self, crypto='ETH-USD', day=1, month=7, year=2017):
        log_print("\ntraining agent...")

        self.agent.reinit(crypto=crypto)
        score = self.agent.start_train_trading(day, month, year)
        print_daily_score(day, score[0], score[1], score[2], score[3])

        s = "run_"+str(year)+"_"+str(month)+"_"+str(day)
        plot_candlestick(s, self.agent.quant.get_data_history(80), self.agent.quant.get_signal_history(80))



    def validate_on_day(self, crypto='ETH-USD', day=1, month=7, year=2017):
        self.agent.reinit(crypto=crypto)
        base_score, base_cumulatives = self.run_validation(strategy="MOMENTUM", day=day, month=month, year=year)
        base2_score, base2_cumulatives = self.run_validation(strategy="BUYNHOLD", day=day, month=month, year=year)
        train_score, train_cumulatives = self.run_validation(strategy="AGENT", day=day, month=month, year=year, plot=True)

        s = "validation_"+crypto+"_"+str(year)+"_"+str(month)+"_"+str(day)
        plot_cumulative_gains(s, train_cumulatives, base_cumulatives, base2_cumulatives)



    def run_baseline(self, strategy="MOMENTUM", month=7, year=2017):
        log_print("\nstarting "+strategy+" on month ...")
        scores = []

        for day in range(1, self.max_days+1):
            score = self.agent.start_baseline_trading(strategy=strategy, day=day, month=month, year=year)
            print_daily_score(day, score[0], score[1], score[2], score[3])
            scores.append(score)

        return scores


    def run_validation(self, strategy="AGENT", day=1, month=7, year=2017, plot=False):
        log_print("\nvalidating "+strategy+" ...")

        if strategy == "AGENT":
            score = self.agent.start_trading(day, month, year)
            print_daily_score(day, score[0], score[1], score[2], score[3])
        else:
            score = self.agent.start_baseline_trading(strategy, day, month, year)
            print_daily_score(day, score[0], score[1], score[2], score[3])

        if plot:
            s = "validation_"+str(year)+"_"+str(month)+"_"+str(day)
            plot_returns_variation(s, self.agent.quant.get_data_history(80))
            plot_bollinger_bands(s, self.agent.quant.get_data_history(80))

        return [score, self.agent.quant.get_cumulative_returns()]


    def unpack_scores(self, scores):
        scores = np.array(scores)
        p = sum(scores[:,0])/self.max_days
        s = sum(scores[:,1])/self.max_days
        d = sum(scores[:,2])/self.max_days
        e = sum(scores[:,3])/self.max_days
        return [p, s, d, e]


    def overview_month(self, product_id='ETH-USD', month=7, year=2017):
        plt.clf()
        q = Quant()
        b = Broker(product_id)
        b.granularity = 21600 # six hours

        log_print("\ngetting month data ...")
        for day in range(1,28):
            log_print("day "+str(day)+" data ...")
            b.set_date(day, month, year)
            done, s = b.get_data()
            q.add_transition(s, 1, 0, done)
            while not done:
                q.add_transition(s,1, 0, done)
                done, s = b.get_data()
            time.sleep(0.03)

        s = "month_overview_"+product_id+"_"+str(year)+"_"+str(month)
        plot_candlestick(s, q.get_data_history())
        plot_bollinger_bands(s, q.get_data_history())
        plot_returns_variation(s, q.get_data_history())


    def test_on_synthetic_data(self, epochs=5):
        self.agent.reinit(epochs=epochs, max_days=1)

        log_print("\nstarting trainning on synthetic data ...")
        for i in range(epochs):
            self.agent.broker.generate_synthetic_data()
            score = self.agent.start_train_trading()

            s = "synthetic_data_"+str(i+1)
            plot_candlestick(s, self.agent.quant.get_data_history(100), self.agent.quant.get_signal_history(100))


