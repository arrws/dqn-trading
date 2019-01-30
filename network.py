#!/usr/bin/python
# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np


class Network:

    def __init__(self, input_size, output_size, nntype="SIMPLE", network=None, verbose=True):
        self.verbose = verbose
        self.nntype = nntype

        if self.nntype == "LSTM":
            self.init_lstm_network(input_size=input_size, output_size=output_size, lstm_size=32, num_layers=1, keep_prob=0.5, network=network)
        elif self.nntype == "DLSTM":
            self.init_lstm_network(input_size=input_size, output_size=output_size, lstm_size=16, num_layers=2, keep_prob=0.5, network=network)
        else: # SIMPLE
            self.init_simple_network(input_size=input_size, output_size=output_size, num_neurons=32, network=network)


    def init_simple_network(self, input_size, output_size, num_neurons=32, network=None):
        # WEIGHTS
        self.W1 = self.weight_variable([input_size, num_neurons])
        self.b1 = self.bias_variable([num_neurons])

        self.W2 = self.weight_variable([num_neurons, output_size])
        self.b2 = self.bias_variable([output_size])

        # INPUT
        self.x = tf.placeholder(tf.float32, [None, input_size])
        self.a = tf.placeholder(tf.float32, [None, output_size])
        self.y = tf.placeholder(tf.float32, [None])

        # LAYERS
        h1 = tf.nn.relu(tf.matmul(self.x, self.W1) + self.b1)
        self.q = tf.matmul(h1, self.W2) + self.b2

        # gradient step
        self.q_value = tf.reduce_sum(tf.multiply(self.q, self.a), reduction_indices=1)
        self.loss = tf.reduce_mean(tf.square(self.y - self.q_value))

        global_step = tf.Variable(0, name='global_step', trainable=False)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(self.loss)

        self.saver = tf.train.Saver(max_to_keep = 3)


    def init_lstm_network(self, input_size, output_size, lstm_size=32, num_layers=1, keep_prob=0.5, batch_size=1, network=None):

        # WEIGHTS
        self.W = self.weight_variable([lstm_size, output_size])
        self.b = self.bias_variable([output_size])

        # INPUT
        self.x = tf.placeholder(tf.float32, [None, input_size])

        self.a = tf.placeholder(tf.float32, [None, output_size])
        self.y = tf.placeholder(tf.float32, [None])

        # LAYERS
        self.cell = tf.contrib.rnn.MultiRNNCell([self.lstm_cell(lstm_size, keep_prob) for _ in range(num_layers)])
        initial_state = self.cell.zero_state(batch_size, tf.float32)

        outputs, final_state = tf.nn.dynamic_rnn(self.cell, tf.expand_dims(self.x, axis=0), initial_state=initial_state)

        self.q = tf.matmul(outputs[-1], self.W) + self.b

        # gradient step
        self.q_value = tf.reduce_sum(tf.multiply(self.q, self.a), reduction_indices=1)
        self.loss = tf.reduce_mean(tf.square(self.y - self.q_value))

        global_step = tf.Variable(0, name='global_step', trainable=False)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.00025).minimize(self.loss)

        self.saver = tf.train.Saver(max_to_keep = 3)


    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.01)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.01, shape=shape)
        return tf.Variable(initial)

    def lstm_cell(self, lstm_size, keep_prob):
        return tf.contrib.rnn.DropoutWrapper( tf.contrib.rnn.BasicLSTMCell(lstm_size), output_keep_prob=keep_prob)

    def evaluate(self, sess, data):
        return self.q.eval(session=sess, feed_dict={self.x: data})

    def train( self, sess, actions, data, target):
        sess.run(self.optimizer, feed_dict={self.a: actions, self.x: data, self.y: target})


