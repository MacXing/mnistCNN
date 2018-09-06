# -*- coding: utf-8 -*- 
# @Time : 2018/9/5 14:46 
# @Author : Allen 
# @Site :  训练文件
import configparser
from mnsitCnn import MnistCNN
import tensorflow as tf
import os
import data_helper


def train():
    '''
        para
    '''
    cf = configparser.ConfigParser()
    cf.read('conf.ini', encoding='utf-8-sig')
    height = int(cf.get('para', 'height'))
    width = int(cf.get('para', 'width'))
    keep_prob = float(cf.get('para', 'keep_prob'))
    num_class = int(cf.get('para', 'num_class'))
    learning_rate = float(cf.get('para', 'learning_rate'))
    epochs = int(cf.get('para', 'epochs'))
    display_step = int(cf.get('para', 'display_step'))
    batch_size = int(cf.get('para', 'batch_size'))
    model_path = os.path.join(os.getcwd(), cf.get('para', 'model_path'))
    summaries_path = os.path.join(os.getcwd(), cf.get('para', 'summaries_path'))
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    if not os.path.exists(summaries_path):
        os.makedirs(summaries_path)
    '''
        data
    '''
    mnist = data_helper.get_mnist()
    '''
        model
    '''

    with tf.Graph().as_default():
        x = tf.placeholder(tf.float32, [None, height * width], name='input')
        labels = tf.placeholder(tf.float32, [None, num_class], name='label')
        dropout_keep_prob = tf.placeholder(tf.float32)
        m_cnn = MnistCNN(height, width, num_class, learning_rate)
        predicts = m_cnn.cnn_graph(x, dropout_keep_prob)

        loss = m_cnn.loss_graph(predicts, labels)
        optimizer = m_cnn.optimizer_graph(loss)
        accuracy = m_cnn.accuracy_graph(predicts, labels)
        summary_op = m_cnn.summary_op()
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            summary_writer = tf.summary.FileWriter(summaries_path, sess.graph)
            for epoch in range(epochs):
                avg_accuracy = 0
                # total_batch = int(mnist.train.num_examples / batch_size)
                total_batch = 50
                for i in range(total_batch):
                    batch_train_x, batch_train_y = mnist.train.next_batch(batch_size)
                    _, _loss, _, summary = sess.run([predicts, loss, optimizer, summary_op], feed_dict={
                        x: batch_train_x,
                        labels: batch_train_y,
                        dropout_keep_prob: keep_prob,
                    })
                    summary_writer.add_summary(summary, epoch)
                    avg_accuracy += sess.run(accuracy, feed_dict={
                        x: batch_train_x,
                        labels: batch_train_y,
                        dropout_keep_prob: 1,
                    }) / total_batch
                if epoch % display_step == 0:
                    batch_test_x = mnist.test.images[:500]
                    batch_test_y = mnist.test.labels[:500]
                    _acc = sess.run(accuracy, feed_dict={
                        x: batch_test_x,
                        labels: batch_test_y,
                        dropout_keep_prob: 1,
                    })
                    saver.save(sess, model_path + 'model.ckpt')
                    print("epoch:{}/{},train_acc:{},test_acc:{}".format(epoch, epochs, avg_accuracy, _acc))


if __name__ == '__main__':
    train()
