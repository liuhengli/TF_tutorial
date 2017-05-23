# encoding: utf-8


"""
@version: ??
@author: Liu Hengli
@license: Apache Licence 
@contact: liuhengli@gmail.com
@site: https://github.com/liuhengli
@software: PyCharm
@file: utlis.py
@time: 2017/5/22 19:07
工具模块
"""
import tensorflow as tf
import numpy as np

def accuracy(logits, labels):
    """
    对模型进行精度估计，来评价模型的好坏
    :param logits: 预测的结果 [batch_size, NUM_CLASSES].
    :param labels: 真实结果
    :return:
        精度，float
    """
    with tf.name_scope('accuracy') as scope:
        correct = tf.equal(tf.argmax(logits, 1), tf.arg_max(labels, 1))
        correct = tf.cast(correct, tf.float32)
        acc = tf.reduce_mean(correct) * 100.0
        tf.summary.scalar(scope + './accuracy', acc)

    return acc

def num_correct_prediction(logits, labels):
  """
  统计预测结果
  Return:
      正确的预测结果数
  """
  correct = tf.equal(tf.arg_max(logits, 1), tf.arg_max(labels, 1))
  correct = tf.cast(correct, tf.int32)
  n_correct = tf.reduce_sum(correct)
  return n_correct

def optimize(loss, learning_rate, global_step):
    """
    选择误差最小化的优化方法,这里默认使用梯度下降
    :param loss: 误差值
    :param learning_rate: 学习率
    :param global_step:
    :return:
    """
    with tf.name_scope('optimizer') as scope:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate,)
        train_op = optimizer.minimize(loss, global_step=global_step)

    return train_op

def load_weigths_biases(data_path, session):
    """加载参数数据"""
    data_dict = np.load(data_path, encoding = 'bytes').item()

    keys = sorted(data_dict.keys())
    for key in keys:
        with tf.variable_scope(key, reuse=True):
            for subkey, data in zip(('weights', 'biases'), data_dict[key]):
                session.run(tf.get_variable(subkey).assign(data))

def test_load():
    """测试加载参数功能"""
    data_path = './/vgg16_pretrain//vgg16.npy'

    data_dict = np.load(data_path, encoding='latin1').item()
    keys = sorted(data_dict.keys())
    for key in keys:
        weights = data_dict[key][0]
        biases = data_dict[key][1]
        print('\n')
        print(key)
        print('weights shape: ', weights.shape)
        print('biases shape: ', biases.shape)

def load_with_skip(data_path, session, skip_layer):
    data_dict = np.load(data_path, encoding='latin1').item()
    for key in data_dict:
        if key not in skip_layer:
            with tf.variable_scope(key, reuse=True):
                for subkey, data in zip(('weights', 'biases'), data_dict[key]):
                    session.run(tf.get_variable(subkey).assign(data))

def print_all_variable(train_only=True):
    """
    打印所有的variables
    :param train_only: 如果为True就只打印trainable variables
    :return:
        no return
    """
    if train_only:
        vars = tf.trainable_variables()
        print(" [*] print trainable varlables")
    else:
        try:
            vars = tf.global_variables()
        except:
            vars = tf.all_variables()
        print(" [*] print global variables")
    for idx, v in enumerate(vars):
        print(" var {:3}: {:15} {}".format(idx, str(v.get_shape()), v.name))

def init_weights(kernel_shape, is_uniform = True):
    """
    初始化weights
    :param kernel_shape: weights的大小
    :param is_uniform:  boolen type.
                if True: use uniform distribution initializer
                if False: use normal distribution initizalizer
    :return:
        weight tensor
    """
    w = tf.get_variable(name='weights',
                        shape=kernel_shape,
                        initializer=tf.contrib.layers.xavier_initializer())
    return w

def init_biases(bias_shape):
    """
   初始化biases
   :param bias_shape: 1D tensor
   :return:
        1D tensor
    """
    b = tf.get_variable(name='biases',
                        shape=bias_shape,
                        initializer=tf.constant_initializer(0.0))
    return b

# data_dict = {'1': (1, 2), '2':(3, 4)}
# x = zip(('weights', 'biases'), (1, 2))
# for i in x:
#     print(i)
