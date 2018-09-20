#!/usr/bin/env python3
# -*- coding:utf-8 -*- 
# Author: lionel

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

from src.siamese_model import siamese_model
from tensorflow.contrib import lookup

flags = tf.app.flags
flags.DEFINE_integer("sentence_max_len", 10, "max length of sentences")
flags.DEFINE_string("pad_word", "<pad>", "used for pad sentence")
flags.DEFINE_string("train_data", "../data/train.csv", "train data")
flags.DEFINE_string("valid_data", "../data/valid.csv", "valid data")
flags.DEFINE_string("vocab_data", "../data/vocab.csv", "vocab file")
flags.DEFINE_string("tensorboard_dir", "../tensorboard", 'tensorboard file')
flags.DEFINE_integer("batch_size", 120, "batch size")
flags.DEFINE_integer("embedding_size", 30, "embedding size")
flags.DEFINE_integer("train_data_size", 18000, "train_data_size")
flags.DEFINE_integer("valid_data_size", 2000, "train_data_size")
FLAGS = flags.FLAGS


def parse_line(line, vocab):
    def get_content(record):
        fields = record.decode('utf-8').strip().split("\t")
        if len(fields) != 3:
            raise ValueError("invalid record %s" % record)
        # words = [ele for ele in list(extract_chinese_word(fields[0]))]
        # words2 = [ele for ele in list(extract_chinese_word(fields[1]))]
        words = [ele for ele in list(fields[0])]
        words2 = [ele for ele in list(fields[1])]
        if len(words) > FLAGS.sentence_max_len:
            words = words[:FLAGS.sentence_max_len]
        if len(words) < FLAGS.sentence_max_len:
            for i in range(FLAGS.sentence_max_len - len(words)):
                words.insert(0, FLAGS.pad_word)
        if len(words2) > FLAGS.sentence_max_len:
            words2 = words2[:FLAGS.sentence_max_len]
        if len(words2) < FLAGS.sentence_max_len:
            for i in range(FLAGS.sentence_max_len - len(words2)):
                words2.insert(0, FLAGS.pad_word)
        # 如果使用 tf1.4版本 需要加上下面这两条代码
        # words = [ele.encode('utf-8') for ele in words]
        # words2 = [ele.encode('utf-8') for ele in words2]
        return [words, words2, np.float32(fields[2])]

    result = tf.py_func(get_content, [line], [tf.string, tf.string, tf.float32])
    result[0].set_shape([FLAGS.sentence_max_len])
    result[1].set_shape([FLAGS.sentence_max_len])
    result[2].set_shape([])
    ids = vocab.lookup(result[0])
    ids2 = vocab.lookup(result[1])
    return {"sentence": ids, "sentence2": ids2}, result[2]


def input_fn(path_csv, path_vocab, shuffle_buffer_size, num_oov_buckets):
    vocab = lookup.index_table_from_file(path_vocab, num_oov_buckets=num_oov_buckets, default_value=1)
    dataset = tf.data.TextLineDataset(path_csv)
    dataset = dataset.map(lambda line: parse_line(line, vocab))
    if shuffle_buffer_size > 0:
        dataset = dataset.shuffle(shuffle_buffer_size).repeat()
        dataset = dataset.batch(FLAGS.batch_size).prefetch(1)
    else:
        dataset = dataset.batch(FLAGS.batch_size).prefetch(1)
    return dataset


def evaluate(sess, valid_dataset):
    valid_iterator = valid_dataset.make_initializable_iterator()
    sess.run(valid_iterator.initializer)
    next_element = valid_iterator.get_next()
    valid_loss, valid_accuracy = 0.0, 0.0
    try:
        while True:
            features, labels = sess.run(next_element)
            feed_dict = {train_model.sentence: features['sentence'],
                         train_model.sentence2: features['sentence2'],
                         train_model.labels: labels}
            valid_loss += sess.run(train_model.loss, feed_dict=feed_dict) * len(labels)
            valid_accuracy += sess.run(train_model.accuracy, feed_dict=feed_dict) * len(labels)
    except tf.errors.OutOfRangeError:
        pass

    valid_loss = valid_loss / FLAGS.valid_data_size
    valid_accuracy = valid_accuracy / FLAGS.valid_data_size

    return valid_loss, valid_accuracy


def train():
    tf.summary.scalar("loss", train_model.loss)
    tf.summary.scalar("accuracy", train_model.accuracy)
    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter(FLAGS.tensorboard_dir)

    train_dataset = input_fn(FLAGS.train_data, FLAGS.vocab_data, shuffle_buffer_size=FLAGS.train_data_size,
                             num_oov_buckets=0)
    valid_dataset = input_fn(FLAGS.valid_data, FLAGS.vocab_data, shuffle_buffer_size=0, num_oov_buckets=0)

    sess = tf.Session()
    tf.tables_initializer().run(session=sess)
    train_iterator = train_dataset.make_initializable_iterator()
    sess.run(train_iterator.initializer)
    sess.run(tf.global_variables_initializer())
    writer.add_graph(sess.graph)
    best_valid_accuracy = 0.0
    train_next = train_iterator.get_next()
    # batch = int(100 * math.ceil(FLAGS.train_data_size / FLAGS.batch_size))
    for i in range(5000):
        features, labels = sess.run(train_next)
        feed_dict = {train_model.sentence: features['sentence'],
                     train_model.sentence2: features['sentence2'],
                     train_model.labels: labels}
        loss, accuracy = sess.run([train_model.loss, train_model.accuracy], feed_dict=feed_dict)
        if i % 20 == 0:
            valid_loss, valid_accuracy = evaluate(sess, valid_dataset)

            if best_valid_accuracy < valid_accuracy:
                best_valid_accuracy = valid_accuracy
                s = sess.run(merged_summary, feed_dict=feed_dict)
                writer.add_summary(s, i)
                print("Iter: %d, train_loss: %f, train_accuracy: %f" % (i, loss, accuracy))
                print("Iter: %d, valid_loss: %f, valid_accuracy: %f\n\n" % (i, valid_loss, valid_accuracy))
        sess.run(train_model.optimizer, feed_dict=feed_dict)


if __name__ == "__main__":
    params = {
        'vocab_size': 12000,
        'embedding_size': 100,
        'sequence_length': 10,
        'learning_rate': 0.01,
        'n_layers': 3,
        'n_hidden': 100,
        'dropout': 0.8
    }
    train_model = siamese_model(params)
    train()