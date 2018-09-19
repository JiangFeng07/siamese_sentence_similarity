#!/usr/bin/env python3
# -*- coding:utf-8 -*- 
# Author: lionel


import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib.layers import xavier_initializer


class siamese_model(object):
    def __init__(self, params):
        # self.iterator = iterator
        # features, labels = self.iterator.get_next()
        # self.sentence = features['sentence']
        # self.sentence2 = features['sentence2']
        # self.labels = labels

        self.sentence = tf.placeholder(tf.int32, shape=[None, params['sequence_length']], name='sentence')
        self.sentence2 = tf.placeholder(tf.int32, shape=[None, params['sequence_length']], name='sentence2')
        self.labels = tf.placeholder(tf.float32, shape=[None], name='labels')

        with tf.name_scope('embedding'):
            embeddings = tf.get_variable(name='embeddings', dtype=tf.float32,
                                         shape=[params['vocab_size'], params['embedding_size']],
                                         initializer=xavier_initializer())
            self.sentence_embed = tf.nn.embedding_lookup(embeddings, self.sentence)
            self.sentence2_embed = tf.nn.embedding_lookup(embeddings, self.sentence2)

        inputs = tf.transpose(self.sentence_embed, [1, 0, 2])
        inputs = tf.reshape(inputs, [-1, params['n_hidden']])
        inputs = tf.split(inputs, params['sequence_length'], 0)

        inputs2 = tf.transpose(self.sentence2_embed, [1, 0, 2])
        inputs2 = tf.reshape(inputs2, [-1, params['n_hidden']])
        inputs2 = tf.split(inputs2, params['sequence_length'], 0)

        with tf.name_scope('out'):
            out_first = self.biRnn(inputs, params, scope='first')
            out_second = self.biRnn(inputs2, params, scope='second')
            distance = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(out_first, out_second)), 1, keepdims=True))
            distance = tf.div(distance, tf.add(tf.sqrt(tf.reduce_sum(tf.square(out_first), 1, keepdims=True)),
                                               tf.sqrt(tf.reduce_sum(tf.square(out_second), 1, keepdims=True))))
            distance = tf.reshape(distance, [-1], name='distance')
        with tf.name_scope('loss'):
            self.loss = self.constrastive_loss(self.labels, distance)

        self.optimizer = tf.train.AdadeltaOptimizer(learning_rate=params['learning_rate']).minimize(self.loss)

        with tf.name_scope('accuracy'):
            temp_sim = tf.subtract(tf.ones_like(distance), tf.rint(distance),
                                   name="temp_sim")  # auto threshold 0.5
            correct_predictions = tf.equal(temp_sim, self.labels)
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32), name="accuracy")

    def constrastive_loss(self, y, d):
        N = tf.cast(tf.shape(y)[0], tf.float32)
        part = y * tf.square(d)
        part2 = (1 - y) * tf.square(tf.maximum(1 - d, 0))
        return tf.reduce_sum(part + part2) / (2.0 * N)

    def biRnn(self, x, params, scope):
        with tf.name_scope("fw_" + scope), tf.variable_scope("fw_" + scope):
            lstm_fw_cell_list = []
            for _ in range(params['n_layers']):
                fw_cell = tf.nn.rnn_cell.BasicLSTMCell(params['n_hidden'], forget_bias=1.0, state_is_tuple=True)
                fw_cell_drop = rnn.DropoutWrapper(fw_cell, output_keep_prob=params['dropout'])
                lstm_fw_cell_list.append(fw_cell_drop)
            lstm_fw_cell_m = tf.nn.rnn_cell.MultiRNNCell(cells=lstm_fw_cell_list, state_is_tuple=True)
        with tf.name_scope('bw_' + scope), tf.variable_scope('bw_' + scope):
            lstm_bw_cell_list = []
            for _ in range(params['n_layers']):
                bw_cell = tf.nn.rnn_cell.BasicLSTMCell(params['n_hidden'], forget_bias=1.0, state_is_tuple=True)
                bw_cell_drop = rnn.DropoutWrapper(bw_cell, output_keep_prob=params['dropout'])
                lstm_bw_cell_list.append(bw_cell_drop)
            lstm_bw_cell_m = tf.nn.rnn_cell.MultiRNNCell(cells=lstm_bw_cell_list, state_is_tuple=True)

        with tf.name_scope('bw_' + scope), tf.variable_scope('bw_' + scope):
            outputs, _, _ = tf.nn.static_bidirectional_rnn(lstm_fw_cell_m, lstm_bw_cell_m, x, dtype=tf.float32)
        return outputs[-1]
