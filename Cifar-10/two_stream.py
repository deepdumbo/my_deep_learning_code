# -*- coding:utf-8 -*-
import pickle as p
import numpy as np
import tensorflow as tf


def load_CIFAR_batch(filename):
    """ load single batch of cifar """
    with open(filename, 'rb')as f:
        datadict = p.load(f, encoding='iso-8859-1')
        X = datadict['data']
        Y = datadict['labels']
        X = X.reshape(10000, 3, 32, 32)
        Y = np.array(Y)
        return X, Y


def load_CIFAR(Foldername):
    train_data = np.zeros([50000, 32, 32, 3])
    train_label = np.zeros([50000, 10])
    for sample in range(5):
        X, Y = load_CIFAR_batch(Foldername + "/data_batch_" + str(sample + 1))

        for i in range(3):
            train_data[10000 * sample:10000 * (sample + 1), :, :, i] = X[:, i, :, :]
        for i in range(10000):
            train_label[i + 10000 * sample][Y[i]] = 1

    test_data = np.zeros([10000, 32, 32, 3])
    test_label = np.zeros([10000, 10])
    X, Y = load_CIFAR_batch(Foldername + "/test_batch")
    for i in range(3):
        test_data[0:10000, :, :, i] = X[:, i, :, :]
    for i in range(10000):
        test_label[i][Y[i]] = 1

    return train_data, train_label, test_data, test_label


def data_shullfe(train_data, train_label, test_data, test_label):
    seed = np.random.randint(10000)
    np.random.seed(seed)
    np.random.shuffle(train_data)
    np.random.seed(seed)
    np.random.shuffle(train_label)

    seed = np.random.randint(10000)
    np.random.seed(seed)
    np.random.shuffle(test_data)
    np.random.seed(seed)
    np.random.shuffle(test_label)
    return train_data, train_label, test_data, test_label


def max_pooling_3d(input, depth, width, height):
    return tf.nn.max_pool3d(input, ksize=[1, depth, width, height, 1], strides=[1, depth, width, height, 1],
                            padding='SAME')


def max_pooling_2d(input, width, height):
    return tf.nn.max_pool(input, ksize=[1, width, height, 1], strides=[1, width, height, 1], padding='SAME')


def conv3d(input, name, depth, kernel_size, input_channel, output_channel, depth_strides=1):
    W = tf.get_variable(name=name + '_Weight', shape=[depth, kernel_size, kernel_size, input_channel, output_channel])
    b = tf.get_variable(name=name + '_bias', shape=[output_channel])
    return tf.add(tf.nn.conv3d(input, W, strides=[1, depth_strides, 1, 1, 1], padding='SAME'), b)


def conv2d(input, name, kernel_size, input_channel, output_channel):
    W = tf.get_variable(name=name + '_Weight', shape=[kernel_size, kernel_size, input_channel, output_channel])
    b = tf.get_variable(name=name + '_bias', shape=[output_channel])
    return tf.add(tf.nn.conv2d(input, W, strides=[1, 1, 1, 1], padding='SAME'), b)


def fc(input, name, input_channel, output_channel):
    W = tf.get_variable(name=name + '_Weight', shape=[input_channel, output_channel])
    b = tf.get_variable(name=name + '_bias', shape=[output_channel])
    return tf.matmul(input, W) + b


def batch_norm(input, name, train, decay=0.9):
    shape = input.get_shape()
    gamma = tf.get_variable(name=name + "_gamma", shape=[shape[-1]],
                            initializer=tf.random_normal_initializer(1.0, 0.01))
    beta = tf.get_variable(name=name + "_beta", shape=[shape[-1]], initializer=tf.constant_initializer(0.0))

    batch_mean, batch_variance = tf.nn.moments(input, list(range(len(shape) - 1)))

    moving_mean = tf.get_variable(name=name + '_moving_mean', shape=[shape[-1]], initializer=tf.zeros_initializer(),
                                  trainable=False)
    moving_variance = tf.get_variable(name=name + '_moving_variance', shape=[shape[-1]],
                                      initializer=tf.ones_initializer(), trainable=False)

    def train_data():
        apply_mean = tf.assign(moving_mean, moving_mean * decay + batch_mean * (1 - decay))
        apply_variance = tf.assign(moving_variance, moving_variance * decay + batch_variance * (1 - decay))
        with tf.control_dependencies([apply_mean, apply_variance]):
            return tf.identity(batch_mean), tf.identity(batch_variance)

    def test_data():
        return moving_mean, moving_variance

    # tf.cond() is a stupid function
    # do not change any code followed
    # or you will fix it for a week
    #
    # 2018.1.27 20:39

    mean, variance = tf.cond(tf.equal(train, tf.constant(True)), train_data, test_data)

    result = tf.nn.batch_normalization(x=input, mean=mean, variance=variance, offset=beta, scale=gamma,
                                       variance_epsilon=1e-5)
    return result


def Expand_dim_down(input, num):
    """
    
    Args:
        input: [a, b]
        num: 2

    Returns: [a, b, 1, 1]

    """
    res = input
    rank = tf.shape(input).get_shape().as_list()[0]
    for i in range(rank, rank + num):
        res = tf.expand_dims(res, i)
    return res


def Expand_dim_up(input, num):
    """
    
    Args:
        input: [a, b]
        num: 2

    Returns: [1, 1, a, b]

    """
    res = input
    for i in range(num):
        res = tf.expand_dims(res, 0)
    return res


def Gabor_filter(input, name, theta_num, lambda_num, size):
    """
    Theta and Lambda is very important in Gabor filter.
    Theta: [-pi, pi]
    Lambda: [2, input_size/5]
    
    Args:
        input: 
        name: 
        theta_num: 
        lambda_num: 
        size: kernel size
        in_channel: 

    Returns:

    """
    shape = input.get_shape().as_list() # [batch, height, width, input_channel]
    input_channel = shape[3]
    output_channel = theta_num*lambda_num

    pi = tf.asin(1.0) * 2.0
    Theta = tf.get_variable(name=name+"_theta", shape=[theta_num],
                            initializer=tf.random_uniform_initializer(minval=-pi, maxval=pi))
    Lambda = tf.get_variable(name=name+"_lambda", shape=[lambda_num],
                            initializer=tf.random_uniform_initializer(minval=2.0, maxval=shape[1]))
    Theta, Lambda = tf.meshgrid(Theta, Lambda)
    Theta = tf.tile(input=Expand_dim_up(tf.reshape(Theta, [-1]), 3), multiples=[size, size, input_channel, 1])
    Lambda = tf.tile(input=Expand_dim_up(tf.reshape(Lambda, [-1]), 3), multiples=[size, size, output_channel, 1])
    # print(Theta.get_shape())

    coordinate_start = -(size - 1) / 2.0
    coordinate_stop = -coordinate_start
    coordinate = tf.linspace(coordinate_start, coordinate_stop, size)
    x, y = tf.meshgrid(coordinate, coordinate)
    x = tf.tile(input=Expand_dim_down(x, 2), multiples=[1, 1, input_channel, output_channel])
    y = tf.tile(input=Expand_dim_down(y, 2), multiples=[1, 1, input_channel, output_channel])
    # print(x.get_shape())

    Sigma = tf.multiply(0.56, Lambda)

    x_ = tf.add(tf.multiply(x, tf.cos(Theta)), tf.multiply(y, tf.sin(Theta)))
    y_ = tf.add(-tf.multiply(x, tf.sin(Theta)), tf.multiply(y, tf.cos(Theta)))
    res = tf.multiply(tf.exp(-tf.div(tf.add(tf.square(x_), tf.square(y_)), tf.multiply(2.0, tf.square(Sigma)))),
                      tf.cos(2.0 * pi * x_ / Lambda))
    return res


print("read data...")
train_data, train_label, test_data, test_label = load_CIFAR("ori_data")
print("finish")


batch_size = 50
x = tf.placeholder("float", shape=[batch_size, 32, 32, 3])
y = tf.placeholder("float", shape=[batch_size, 10])
keep_prob = tf.placeholder("float")
BN_train = tf.placeholder("bool", shape=[])

Gabor_filter(x, 'Gabor', 4, 4, 7)