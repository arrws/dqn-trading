#!/usr/bin/python
# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import sys
import random
import time
import os

from network import *
from broker import *
from quant import *
from utils import *

batch_size = 128 #40
no_features = 8
no_actions = 3

gamma = 0.9

start_epsilon = 1.0  # starting value of epsilon
final_epsilon = 0.01  # final value of epsilon
epsilon = start_epsilon  # current epsilon

#update_network_freq = 100



class Agent:

    def __init__(self, network_type):
        # Neural Network Setup
        self.nn = Network(nntype=network_type, input_size=no_features, output_size=no_actions, network = None)
        self.sess = tf.InteractiveSession()
        self.sess.run(tf.initialize_all_variables())
        #self.restore_network_progress()

        self.quant = Quant()
        self.broker = Broker()

        self.print_freq = 10
        self.reinit()


    def reinit(self, crypto='ETH-USD', epochs=0, max_days=0):
        global epsilon
        if epochs != 0:
            epsilon = start_epsilon
            self.anneal_steps = 288 * max_days * (epochs-5)
        else:
            epsilon = final_epsilon
            self.anneal_steps = 1

        self.global_step = 0
        self.broker.set_crypto(crypto)



    def start_trading(self, day=1, month=7, year=2017, verbose=False):
        self.broker.set_date(day, month, year)
        self.quant.reinit()
        self.step = 0

        done, s = self.broker.get_data()
        self.quant.add_transition(s, 0, 0, done)

        while not done:
            self.step += 1

            a, qmax = self.predict_action(s)
            done, s2 = self.broker.get_data()

            r = self.quant.get_return(a, s2)
            self.quant.add_transition(s2, a, r, done)
            s = s2

            # printing...
            if verbose and self.step % self.print_freq == 0:
                self.print_trade_details(a, r, qmax)

        score = self.quant.eval_performance()
        return score


    def start_train_trading(self, day=1, month=7, year=2017, verbose=False):
        self.broker.set_date(day, month, year)
        self.quant.reinit()
        self.step = 0

        done, s = self.broker.get_data()
        self.quant.add_transition(s, 0, 0, done)

        a1 = []
        a2 = []
        a3 = []
        while not done:
            self.step += 1
            self.global_step += 1

            a, qmax = self.choose_action(s)
            if a==1:
                a1.append(a)
            elif a==2:
                a2.append(a)
            else:
                a3.append(a)

            done, s2 = self.broker.get_data()

            r = self.quant.get_return(a, s2)
            self.quant.add_transition(s2, a, r, done)
            s = s2

            self.anneal_epsilon()
            self.do_train_step()

            # printing...
            if verbose and self.step % self.print_freq == 0:
                self.print_trade_details(a, r, qmax)

        score = self.quant.eval_performance()
        return score


    def start_baseline_trading(self, strategy="MOMENTUM", day=1, month=7, year=2017, verbose=False):
        self.broker.set_date(day, month, year)
        self.quant.reinit()
        self.step = 0

        done, s = self.broker.get_data()
        self.quant.add_transition(s, 0, 0, done)

        while not done:
            self.step += 1

            a = self.quant.get_last_trend()
            if strategy == "BUYNHOLD":
                a = self.quant.get_last_bollinger()

            qmax = 0
            done, s2 = self.broker.get_data()

            r = self.quant.get_return(a, s2)
            self.quant.add_transition(s2, a, r, done)
            s = s2

            # printing...
            if verbose and self.step % self.print_freq == 0:
                self.print_trade_details(a, r, qmax)

        score = self.quant.eval_performance()
        return score



    def anneal_epsilon(self):
        global epsilon
        if epsilon > final_epsilon:  # anneal explorativeness
            epsilon -= (start_epsilon - final_epsilon) / self.anneal_steps  # update epsilon


    def do_train_step(self):
        # sample a minibatch to train on
        s_batch, a_batch, r_batch, s2_batch, d_batch = self.quant.sample_batches(no_actions, batch_size)

        y_batch = []
        q_batch = self.nn.evaluate(self.sess, s2_batch)
        #q_batch = self.target_nn.evaluate(self.sess, s2_batch)

        # build cumulative reward batch
        for i in range(len(s_batch)):
            if d_batch[i]: # if episode is done
                y_batch.append(r_batch[i]) # last reward
            else:
                y_batch.append(r_batch[i] + gamma * np.max(q_batch[i])) # discounted future rewards

        self.nn.train(self.sess, a_batch, s_batch, y_batch)


    def predict_action(self, s):
        s_batch = self.quant.get_last_features()
        q = self.nn.evaluate(self.sess, s_batch)[0]
        action = np.argmax(q)
        qmax = np.max(q)
        return [int(action), qmax]

    def choose_action(self, s):
        global epsilon
        action = 0
        if random.random() <= epsilon:
            action = random.randrange(no_actions)
            qmax = 0
            return [int(action), qmax]
        else:
            return self.predict_action(s)



    def restore_network_progress(self, path=""):
        if path == "":
            path = PATH+"/nn"
        # load network if exists
        checkpoint = tf.train.get_checkpoint_state(path)
        if checkpoint and checkpoint.model_checkpoint_path:
            self.nn.saver.restore(self.sess, checkpoint.model_checkpoint_path)
            print("\nSuccessfully restored:", checkpoint.model_checkpoint_path,"\n")
        else:
            print("\nCould not restore network\n")

    def save_network_progress(self, name="0"):
        save_path = self.nn.saver.save(self.sess, PATH+"/nn/model_"+name+".ckpt", global_step = self.global_step)
        print("Successfully saved network:", save_path, "\n")


    def print_trade_details(self, a, r, q):
        print('step: {:d}\taction: {:.2f}\treward: {:.2f}\tqmax: {:.2f}\tepsilon: {:.2f}'.format(self.step, a, r, q, epsilon))


