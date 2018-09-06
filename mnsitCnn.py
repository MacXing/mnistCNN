# -*- coding: utf-8 -*- 
# @Time : 2018/9/5 14:47 
# @Author : Allen 
# @Site :  构建CNN
import tensorflow as tf


class MnistCNN:
    def __init__(self, height, width, num_class=10, learning_rate=0.001):
        self.height = height
        self.width = width
        self.num_class = num_class
        self.learning_rate = learning_rate

    def weight_variable(self, shape, w_alpha=0.01):
        return tf.Variable(w_alpha * tf.random_normal(shape))

    def bias_variable(self, shape, b_alpha=0.1):
        return tf.Variable(b_alpha * tf.random_normal(shape))

    def cnn_graph(self, _x, dropout_keep_prob):
        with tf.name_scope('input'):
            _x = tf.reshape(_x, [-1, self.height, self.width, 1], name='input')
        with tf.name_scope('conv1'):
            conv1 = tf.nn.conv2d(_x, self.weight_variable([3, 3, 1, 64]), strides=[1, 1, 1, 1], padding='SAME')
            conv1 = tf.nn.relu(conv1 + self.bias_variable([64]))
            conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            conv1 = tf.nn.dropout(conv1, dropout_keep_prob, name='conv1')
        with tf.name_scope('conv2'):
            conv2 = tf.nn.conv2d(conv1, self.weight_variable([3, 3, 64, 128]), strides=[1, 1, 1, 1], padding='SAME')
            conv2 = tf.nn.relu(conv2 + self.bias_variable([128]))
            conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            conv2 = tf.nn.dropout(conv2, dropout_keep_prob, name='conv2')
        # with tf.name_scope('conv3'):
        #     conv3 = tf.nn.conv2d(conv2, self.weight_variable([3, 3, 128, 256]), strides=[1, 1, 1, 1], padding='SAME')
        #     conv3 = tf.nn.relu(conv3 + self.bias_variable([256]))
        #     conv3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        #     conv3 = tf.nn.dropout(conv3, self.dropout_keep_prob, name='conv3')
        with tf.name_scope('dense'):
            w = conv2.get_shape().as_list()
            dense = tf.reshape(conv2, [-1, w[1] * w[2] * w[3]])
            dense = tf.matmul(dense, self.weight_variable([w[1] * w[2] * w[3], 1024])) + self.bias_variable([1024])
            dense = tf.nn.dropout(tf.nn.relu(dense), dropout_keep_prob, name='dense')
        with tf.name_scope('output'):
            predict = tf.matmul(dense, self.weight_variable([1024, self.num_class])) + self.bias_variable(
                [self.num_class])

        return predict

    def loss_graph(self, predicts, labels):
        with tf.name_scope('Loss'):
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predicts, labels=labels), name='Loss')
            tf.summary.scalar('loss', loss)
        return loss

    def optimizer_graph(self, loss):
        with tf.name_scope('optimizer'):
            optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(loss)
        return optimizer

    def accuracy_graph(self, predicts, labels):
        with tf.name_scope('Accuracy'):
            corr = tf.equal(tf.argmax(predicts, 1), tf.argmax(labels, 1))
            acc = tf.reduce_mean(tf.cast(corr, tf.float32), name='Accuracy')
            tf.summary.scalar('accuracy', acc)
        return acc

    def summary_op(self):
        return tf.summary.merge_all()
