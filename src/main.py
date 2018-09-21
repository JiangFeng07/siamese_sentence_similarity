#!/usr/bin/env python3
# -*- coding:utf-8 -*- 
# Author: lionel

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import logging
import math
from src.siamese_model import siamese_model
from tensorflow.contrib import lookup

tf.logging.set_verbosity(tf.logging.INFO)
tf.logging._handler.setFormatter(
    logging.Formatter("%(asctime)s [%(levelname)s] <%(processName)s> (%(threadName)s) %(message)s"))

flags = tf.app.flags
flags.DEFINE_integer("sentence_max_len", 10, "max length of sentences")
flags.DEFINE_string("pad_word", "<pad>", "used for pad sentence")
flags.DEFINE_string("train_data", "../data/train.csv", "train data")
flags.DEFINE_string("valid_data", "../data/valid.csv", "valid data")
flags.DEFINE_string("predict_data", "../data/predict.csv", "predict data")
flags.DEFINE_string("vocab_data", "../data/vocab.csv", "vocab file")
flags.DEFINE_string("tensorboard_dir", "../tensorboard", 'tensorboard file')
flags.DEFINE_string("model_path", "../model/model.ckpt", 'model path')
flags.DEFINE_integer("batch_size", 100, "batch size")
flags.DEFINE_integer("train_data_size", 18000, "train_data_size")
flags.DEFINE_integer("valid_data_size", 2000, "train_data_size")
flags.DEFINE_integer("epoch_size", 3, "epoch_size")
flags.DEFINE_integer("vocab_size", 12000, "vocabulary size")
flags.DEFINE_integer("embedding_size", 100, "embedding size")
flags.DEFINE_integer("learning_rate", 0.01, "learning rate")
flags.DEFINE_integer("n_layers", 3, "layers")
flags.DEFINE_integer("n_hidden", 100, "hidden units")
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
        words = [ele.encode('utf-8') for ele in words]
        words2 = [ele.encode('utf-8') for ele in words2]
        return [words, words2, np.float32(fields[2])]

    result = tf.py_func(get_content, [line], [tf.string, tf.string, tf.float32])
    result[0].set_shape([FLAGS.sentence_max_len])
    result[1].set_shape([FLAGS.sentence_max_len])
    result[2].set_shape([])
    ids = vocab.lookup(result[0])
    ids2 = vocab.lookup(result[1])
    return {"sentence": ids, "sentence2": ids2}, result[2]


def input_fn(path_csv, shuffle_buffer_size, mode, vocab):
    dataset = tf.data.TextLineDataset(path_csv)
    dataset = dataset.map(lambda line: parse_line(line, vocab))
    if mode == 'train':
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
                         train_model.labels: labels,
                         train_model.keep_prob: 1.0}
            valid_loss += sess.run(train_model.loss, feed_dict=feed_dict) * len(labels)
            valid_accuracy += sess.run(train_model.accuracy, feed_dict=feed_dict) * len(labels)
    except tf.errors.OutOfRangeError:
        pass

    valid_loss = valid_loss / FLAGS.valid_data_size
    valid_accuracy = valid_accuracy / FLAGS.valid_data_size

    return valid_loss, valid_accuracy


def predict(dataset):
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess=sess, save_path=FLAGS.model_path)

    tf.tables_initializer().run(session=sess)
    iterator = dataset.make_initializable_iterator()
    sess.run(iterator.initializer)
    next_element = iterator.get_next()

    try:
        while True:
            features, labels = sess.run(next_element)
            feed_dict = {train_model.sentence: features['sentence'],
                         train_model.sentence2: features['sentence2'],
                         train_model.keep_prob: 1.0}
            print(sess.run(train_model.predict, feed_dict=feed_dict))
    except tf.errors.OutOfRangeError:
        pass


def train():
    tf.summary.scalar("loss", train_model.loss)
    tf.summary.scalar("accuracy", train_model.accuracy)
    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter(FLAGS.tensorboard_dir)
    saver = tf.train.Saver()

    train_dataset = input_fn(FLAGS.train_data, shuffle_buffer_size=FLAGS.train_data_size, mode='train', vocab=vocab)
    valid_dataset = input_fn(FLAGS.valid_data, shuffle_buffer_size=0, mode='valid', vocab=vocab)

    ## GPU 使用动态增长
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    tf.tables_initializer().run(session=sess)
    train_iterator = train_dataset.make_initializable_iterator()
    sess.run(train_iterator.initializer)
    sess.run(tf.global_variables_initializer())
    writer.add_graph(sess.graph)
    best_valid_accuracy = 0.0
    train_next = train_iterator.get_next()
    batch = int(FLAGS.epoch_size * math.ceil(FLAGS.train_data_size / FLAGS.batch_size))
    for i in range(batch):
        features, labels = sess.run(train_next)
        feed_dict = {train_model.sentence: features['sentence'],
                     train_model.sentence2: features['sentence2'],
                     train_model.labels: labels,
                     train_model.keep_prob: 0.8}
        loss, accuracy = sess.run([train_model.loss, train_model.accuracy], feed_dict=feed_dict)
        if i % 20 == 0:
            valid_loss, valid_accuracy = evaluate(sess, valid_dataset)
            if best_valid_accuracy < valid_accuracy:
                best_valid_accuracy = valid_accuracy
                s = sess.run(merged_summary, feed_dict=feed_dict)
                writer.add_summary(s, i)
                tf.logging.info("Iter: %d, train_loss: %f, train_accuracy: %f" % (i, loss, accuracy))
                tf.logging.info("Iter: %d, valid_loss: %f, valid_accuracy: %f\n\n" % (i, valid_loss, valid_accuracy))
        sess.run(train_model.optimizer, feed_dict=feed_dict)


if __name__ == "__main__":
    params = {
        'vocab_size': FLAGS.vocab_size,
        'embedding_size': FLAGS.embedding_size,
        'sequence_length': FLAGS.sentence_max_len,
        'learning_rate': FLAGS.learning_rate,
        'n_layers': FLAGS.n_layers,
        'n_hidden': FLAGS.n_hidden
    }
    train_model = siamese_model(params)
    vocab = lookup.index_table_from_file(FLAGS.vocab_data, num_oov_buckets=0, default_value=1)

    # predict_dataset = input_fn(FLAGS.predict_data, shuffle_buffer_size=0, mode='predict', vocab=vocab)
    train()
    # predict(predict_dataset)
