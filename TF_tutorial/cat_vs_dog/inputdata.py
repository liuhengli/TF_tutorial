# encoding: utf-8

"""
@version: ??
@author: Liu Hengli
@license: Apache Licence 
@contact: liuhengli@gmail.com
@site: https://github.com/liuhengli
@software: PyCharm
@file: inputdata.py
@time: 2017/5/18 13:22
该文件主要是对数据集进行处理，用于模型的训练
"""

import tensorflow as tf
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt

def get_files(file_dir):
    """
    :param file_dir: 数据文件的路径
    :return: 数据与标签列表
    """
    cats = []
    cat_labels = []
    dogs = []
    dog_labels = []
    #遍历数据文件，制作标签和数据list
    for file in os.listdir(file_dir):
        name = file.split(sep='.')
        if name[0] == 'cat':
            cats.append(file_dir + file)
            cat_labels.append(0)
        else:
            dogs.append(file_dir + file)
            dog_labels.append(1)

    print('cats: %d\ndogs: %d' %(len(cats), len(dogs)))

    data_list = np.hstack((cats, dogs))
    label_list = np.hstack((cat_labels, dog_labels))

    temp = np.array([data_list, label_list])
    # print(temp[0])
    temp = temp.transpose()
    # print(temp[0])
    np.random.shuffle(temp)

    data_list = list(temp[:, 0])
    label_list = list(temp[:, 1])
    label_list = [int(i) for i in label_list]

    return data_list, label_list

def get_batch(image, label, image_W, image_H, batch_size, capacity):
    """
    :param image: 数据路径列表
    :param label: 标签列表
    :param image_W: 图片数据的宽
    :param image_H: 图片数据的高
    :param batch_size: batch size
    :param capacity: 队列的最大容量
    :return:
        image_batch: 4D tensor [batch_size, iamge_W, image_H, 3], dtype=tf.float32
        label_batch: 1D tensor [batch_size], dtype=tf.int32
    """

    #数据类型转换
    image = tf.cast(image, tf.string)
    label = tf.cast(label, tf.int32)

    #生成一个输入队列
    input_queue = tf.train.slice_input_producer([image, label])
    label = input_queue[1]
    image_contents = tf.read_file(input_queue[0])
    image = tf.image.decode_jpeg(image_contents, channels=3)

    # 在此可以进行data argumentation来增加模型的泛化能力/精度等

    #将图片resize到统一的大小
    image = tf.image.resize_image_with_crop_or_pad(image, image_W, image_H)
    #进行标准化
    image = tf.image.per_image_standardization(image)
    #生成batch
    image_batch, label_batch = tf.train.batch([image, label],
                                              batch_size=batch_size,
                                              num_threads=4,
                                              capacity=capacity)

    label_batch = tf.reshape(label_batch, [batch_size])
    image_batch = tf.cast(image_batch, tf.float32)

    return image_batch, label_batch


if __name__ == '__main__':
    #参数设置
    BATCH_SIZE = 2
    CAPACITY = 256
    W = 225
    H = 255
    ratio = 0.2
    file_dir = './cat_vs_dog/train/'

    image_list, label_list = get_files(file_dir)
    image_batch, label_batch = get_batch(image_list, label_list, W, H, BATCH_SIZE, CAPACITY)

    with tf.Session() as sess:
        i = 0
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        try:
            while not coord.should_stop() and i<1:
                img, label = sess.run([image_batch, label_batch])
                #最常见的用例是将某些特殊的操作指定为 "feed" 操作,标记的方法是使用 tf.placeholder() 为这些操作创建占位符.

                for j in np.arange(BATCH_SIZE):
                    print('label: %d' %label[j])
                    plt.imshow(img[j,:, :, :])
                    plt.show()
                i+=1

        except tf.errors.OutOfRangeError:
            print("Done !!!")
        finally:
            coord.request_stop()
        coord.join(threads)



