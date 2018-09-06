# -*- coding: utf-8 -*- 
# @Time : 2018/9/5 14:45 
# @Author : Allen 
# @Site :  主文件，predict
import train
import data_helper
import os
from mnsitCnn import MnistCNN
import tensorflow as tf
import numpy as np


def predict(batch_test_x, model_path):
    x = tf.placeholder(tf.float32, [None, 28 * 28])
    m_cnn = MnistCNN(28, 28)
    dropout_keep_prob = tf.placeholder(tf.float32)
    pre = m_cnn.cnn_graph(x, dropout_keep_prob)
    prediction = tf.argmax(pre, 1)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, os.path.join(model_path, 'model.ckpt'))
        predictions = sess.run(prediction, feed_dict={
            x: batch_test_x,
            dropout_keep_prob: 1,
        })
    return predictions.tolist()


def main():
    model_path = os.path.join(os.getcwd(), 'ckpt', 'checkpoints')
    if not os.path.exists(model_path):
        print("开始训练模型....")
        train.train()
    else:
        if not os.path.getsize(model_path) > 0:
            print("开始训练模型....")
            train.train()
    print("模型训练完成！")
    mnist = data_helper.get_mnist()
    batch_test_x = mnist.test.images[:1000]
    batch_test_y = mnist.test.labels[:1000]
    batch_test_y = list(np.argmax(batch_test_y, 1))
    predictions = predict(batch_test_x, model_path)
    for i in range(len(batch_test_y)):
        print("label:{},predict:{}".format(batch_test_y[i], predictions[i]))


if __name__ == '__main__':
    main()
