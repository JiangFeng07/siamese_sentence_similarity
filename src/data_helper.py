#!/usr/bin/env python3
# -*- coding:utf-8 -*- 
# Author: lionel
import logging

import re
import tensorflow as tf
from random import random
import numpy as np
import collections


class InputHelper(object):
    def load_data(self, filepath):
        logging.info("load data")
        x1 = []
        x2 = []
        y = []
        with tf.gfile.GFile(filepath, 'r') as f:
            for line in f:
                l = self.cleanText(line).strip().split("\t")
                if len(l) < 3:
                    continue
                if random() > 0.5:
                    x1.append(l[0].split(" "))
                    x2.append(l[1].split(" "))
                else:
                    x1.append(l[1].split(" "))
                    x2.append(l[0].split(" "))
                y.append(int(l[2]))
        return np.array(x1), np.array(x2), np.array(y)

    def build_word_dic(self, features, vocab_size=5000):
        word_dic = dict()
        word_dic['pad'] = 0
        word_dic['unk'] = 1
        all_words = []
        for words in features:
            all_words.extend(words)

        counter = collections.Counter(all_words).most_common()
        words, _ = list(zip(*counter))
        for word in words:
            word_dic[word] = len(word_dic)
        # label_set = set(labels)
        # label_dic = dict()
        # for label in label_set:
        #     label_dic[label] = len(label_dic)
        return words, word_dic

    def cleanText(self, s):
        s = re.sub(r"[^\x00-\x7F]+", " ", s)
        s = re.sub(r'[\~\!\`\^\*\{\}\[\]\#\<\>\?\+\=\-\_\(\)]+', "", s)
        s = re.sub(r'( [0-9,\.]+)', r"\1 ", s)
        s = re.sub(r'\$', " $ ", s)
        s = re.sub('[ ]+', ' ', s)
        s = re.sub('[,.]', '', s)
        return s.lower()


if __name__ == "__main__":
    ih = InputHelper()
    x1, x2, y = ih.load_data("../train_snli.txt")
    features = np.concatenate([x1, x2], axis=0)
    print(features[0:10])
    words, word_dic, label_set, label_dic = ih.build_word_dic(features, y)
    print(word_dic)
    print(label_dic)
