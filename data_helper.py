# -*- coding: utf-8 -*- 
# @Time : 2018/9/5 14:45 
# @Author : Allen 
# @Site :  数据处理
from tensorflow.examples.tutorials.mnist.input_data import read_data_sets


def get_mnist():
    return read_data_sets('./data', one_hot=True)

if __name__ == '__main__':
    mnist = get_mnist()
    print(mnist.train.next_batch(1))
