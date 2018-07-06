import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer


class SiameseLSTM(object):
    def __init__(self, vocab_size, embedding_size, hidden_units, layers, batch_size, iterator, learning_rate):
        self.input_x1, self.input_x2, self.input_y = iterator.get_next()
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        with tf.name_scope("embedding"):
            embedding = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0), trainable=True,
                                    name="W")
            # embedding = tf.get_variable(name='embedding', shape=[vocab_size, embedding_size], dtype=tf.float32,
            #                             initializer=xavier_initializer())
            self.embedded_chars1 = tf.nn.embedding_lookup(embedding, self.input_x1)
            self.embedded_chars2 = tf.nn.embedding_lookup(embedding, self.input_x2)

        with tf.name_scope("output"):
            self.out1 = self.BiRNN(self.embedded_chars1, self.dropout_keep_prob, "side1", hidden_units, layers)
            self.out2 = self.BiRNN(self.embedded_chars2, self.dropout_keep_prob, "side2", hidden_units, layers)
            self.distance = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(self.out1, self.out2)), 1, keepdims=True))
            self.distance = tf.div(self.distance,
                                   tf.add(tf.sqrt(tf.reduce_sum(tf.square(self.out1), 1, keepdims=True)),
                                          tf.sqrt(tf.reduce_sum(tf.square(self.out2), 1, keepdims=True))))
            self.distance = tf.reshape(self.distance, [-1], name="distance")

        self.input_y = tf.cast(self.input_y, tf.float32)
        with tf.name_scope("loss"):
            self.loss = self.contrastive_loss(self.input_y, self.distance, batch_size)
        with tf.name_scope("optimizer"):
            self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss)
        with tf.name_scope("accuracy"):
            self.tmp_sim = tf.subtract(tf.ones_like(self.distance), tf.rint(self.distance),
                                       name="tmp_sim")  # auto threshold 0.5
            correct_predictions = tf.equal(self.tmp_sim, self.input_y)
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

    def BiRNN(self, x, dropout, scope, hidden_units, layers):
        # Forward direction cell
        with tf.name_scope("fw" + scope), tf.variable_scope("fw" + scope):
            stacked_rnn_fw = []
            for _ in range(layers):
                fw_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_units)
                lstm_fw_cell = tf.contrib.rnn.DropoutWrapper(fw_cell, output_keep_prob=dropout)
                stacked_rnn_fw.append(lstm_fw_cell)
            lstm_fw_cell_m = tf.nn.rnn_cell.MultiRNNCell(cells=stacked_rnn_fw)

        # Backward direciton cell
        with tf.name_scope("bw" + scope), tf.variable_scope("bw" + scope):
            stacked_rnn_bw = []
            for _ in range(layers):
                bw_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_units)
                lstm_bw_cell = tf.contrib.rnn.DropoutWrapper(bw_cell, output_keep_prob=dropout)
                stacked_rnn_bw.append(lstm_bw_cell)
            lstm_bw_cell_m = tf.nn.rnn_cell.MultiRNNCell(cells=stacked_rnn_bw)

        # Get lstm cell output
        with tf.name_scope("bw" + scope), tf.variable_scope("bw" + scope):
            outputs, output_states = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell_m, lstm_bw_cell_m, x,
                                                                     dtype=tf.float32)
        return tf.concat(outputs, 2)[:, -1, :]

    def contrastive_loss(self, y, d, batch_size):
        tmp = y * tf.square(d)
        # tmp= tf.mul(y,tf.square(d))
        tmp2 = (1 - y) * tf.square(tf.maximum((1 - d), 0))
        return tf.reduce_sum(tmp + tmp2) / batch_size / 2
