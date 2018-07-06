#!/usr/bin/env python3
# -*- coding:utf-8 -*- 
# Author: lionel

import tensorflow as tf

from src.data_helper import InputHelper
from src.siamese_network import SiameseLSTM
import numpy as np

FLAGS = None


def build_dic_hash_table(word_dic):
    word_table = tf.contrib.lookup.HashTable(
        tf.contrib.lookup.KeyValueTensorInitializer(list(word_dic.keys()), list(word_dic.values())), word_dic['unk'])
    return word_table


def gen(x1_text, x2_text, y_label):
    index = 0
    while True:
        y = y_label[index]
        x1 = x1_text[index]
        x2 = x2_text[index]
        yield (x1, x2, y)
        index += 1
        if index == len(x1_text):
            index = 0


def train_input_fn(shuffle_size, batch_size, x1, x2, y, repeat_size=None):
    dataset = tf.data.Dataset.from_generator(lambda: gen(x1, x2, y), (tf.string, tf.string, tf.int32),
                                             (tf.TensorShape([None]), tf.TensorShape([None]), tf.TensorShape([])))
    return dataset.shuffle(shuffle_size).repeat(repeat_size).padded_batch(batch_size=batch_size, padded_shapes=(
        tf.TensorShape([FLAGS.sequence_length]), tf.TensorShape([FLAGS.sequence_length]), tf.TensorShape([])),
                                                                          padding_values=('pad', 'pad', -1))


if __name__ == "__main__":
    tf.flags.DEFINE_string('train_file', '../train.csv', 'Train File path(default: ../train.csv)')
    tf.flags.DEFINE_string('valid_file', '../valid.csv', 'Valid File path(default:../valid.csv)')

    tf.flags.DEFINE_integer("hidden_units", 50, "Hidden Size(default: 50)")
    tf.flags.DEFINE_integer("layers", 3, "Layers (default: 3)")
    tf.flags.DEFINE_float("dropout_keep_prob", 1.0, "Dropout keep probability (default: 1.0)")
    tf.flags.DEFINE_integer("sequence_length", 100, "Sequence Length(default: 50)")
    tf.flags.DEFINE_integer("vocab_size", 40000, "Vocabulary Size(default: 5000)")
    tf.flags.DEFINE_integer("shuffle_size", 5000, "Vocabulary Size(default: 5000)")
    tf.flags.DEFINE_integer("batch_size", 100, "Batch Size (default: 64)")
    tf.flags.DEFINE_integer("embedding_size", 300, "Embedding Size (default: 300)")
    tf.flags.DEFINE_float("learning_rate", 1e-3, "Learning Rate (default: 0.01)")
    FLAGS = tf.flags.FLAGS
    FLAGS.flag_values_dict()

    ih = InputHelper()
    train_x1, train_x2, train_y = ih.load_data(FLAGS.train_file)
    # valid_x1, valid_x2, valid_y = ih.load_data(FLAGS.valid_file)
    features = []
    features.extend(train_x1)
    features.extend(train_x2)
    words, word_dic = ih.build_word_dic(features)
    word_table = build_dic_hash_table(word_dic)
    train_dataset = train_input_fn(FLAGS.shuffle_size, FLAGS.batch_size, train_x1, train_x2, train_y, repeat_size=3)
    train_dataset = train_dataset.map(
        lambda train_x1, train_x2, train_y: (word_table.lookup(train_x1), word_table.lookup(train_x2), train_y))
    train_iterator = train_dataset.make_initializable_iterator()

    # valid_dataset = train_input_fn(len(valid_x1), len(valid_x1), valid_x1, valid_x2, valid_y)
    # valid_dataset = valid_dataset.map(
    #     lambda x1, x2, y: (word_table.lookup(valid_x1), word_table.lookup(valid_x2), valid_y))
    # valid_iterator = valid_dataset.make_initializable_iterator()

    model = SiameseLSTM(FLAGS.vocab_size, FLAGS.embedding_size, FLAGS.hidden_units, FLAGS.layers, FLAGS.batch_size,
                        train_iterator, FLAGS.learning_rate)

    # model2 = SiameseLSTM(FLAGS.vocab_size, FLAGS.embedding_size, FLAGS.hidden_units, FLAGS.layers,
    #                     FLAGS.batch_size,
    #                     valid_iterator, FLAGS.learning_rate)

    with tf.Session() as sess:
        sess.run(train_iterator.initializer)
        # sess.run(valid_iterator.initializer)
        sess.run(tf.tables_initializer())
        sess.run(tf.global_variables_initializer())
        for i in range(1, 10001):
            try:
                if i % 100 == 0:
                    input_x1, input_x2, input_y, pred, distance = sess.run(
                        [model.input_x1, model.input_x2, model.input_y, model.tmp_sim, model.distance],
                        feed_dict={model.dropout_keep_prob: 0.8})

                    accuracy, loss = sess.run([model.accuracy, model.loss], feed_dict={model.dropout_keep_prob: 1.0})
                    print("the %d batch, accuracy is %f, and loss is %f" % (i, accuracy, loss))

                sess.run(model.optimizer, feed_dict={model.dropout_keep_prob: 0.8})

                # if i % 10 == 0 and i != 0:
                #     accuracy, loss = sess.run([model.accuracy, model.loss], feed_dict={model.dropout_keep_prob: 1.0})
                #     print("the %d batch, accuracy is %f, and loss is %f" % (i, accuracy, loss))
            except tf.errors.OutOfRangeError:
                break
