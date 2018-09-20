#!/usr/bin/env python3
# -*- coding:utf-8 -*- 
# Author: lionel

import argparse
import re

import tensorflow as tf

parser = argparse.ArgumentParser()
parser.add_argument('--date_dir', type=str,
                    default="../data/train.csv")
parser.add_argument('--date_dir2', type=str,
                    default="../data/valid.csv")

parser.add_argument('--vocab_dir', type=str,
                    default="../data/vocab.csv")
FLAGS, unparser = parser.parse_known_args()


def build_vocab(date_dir, date_dir2, vocab_dir):
    vocab = set()
    with tf.gfile.GFile(date_dir, 'r') as f:
        f.readline()
        for line in f:
            fields = line.strip().split("\t")
            if len(fields) != 3:
                continue
            sentence = extract_chinese_word(fields[0] + fields[1])
            for ele in sentence:
                vocab.add(ele)

    with tf.gfile.GFile(date_dir2, 'r') as f:
        for line in f:
            fields = line.strip().split("\t")
            if len(fields) != 3:
                continue
            sentence = extract_chinese_word(fields[0] + fields[1])
            for ele in sentence:
                vocab.add(ele)

    vocab_file = tf.gfile.GFile(vocab_dir, 'w')
    for ele in vocab:
        vocab_file.write(str(ele) + "\n")
    vocab_file.close()


def extract_chinese_word(text):
    zh_pattern = re.compile('[^\u4e00-\u9fa5]+')
    return ''.join(zh_pattern.split(text))


if __name__ == "__main__":
    build_vocab(FLAGS.date_dir, FLAGS.date_dir2, FLAGS.vocab_dir)
