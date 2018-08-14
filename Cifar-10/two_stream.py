import tensorflow as tf
import numpy as np
import cv2
import os
import time
from My_dataset_class import MyDataset
os.environ['CUDA_VISIBLE_DEVICES'] = '1'


def max_pooling_3d(input, depth, width, height):
    return tf.nn.max_pool3d(input, ksize=[1, depth, width, height, 1], strides=[1, depth, width, height, 1], padding='SAME')


def max_pooling_2d(input, width, height):
    return tf.nn.max_pool(input, ksize=[1, width, height, 1], strides=[1, width, height, 1], padding='SAME')


def conv3d(input, name, depth, kernel_size, output_channel, depth_strides=1, padding='SAME'):
    input_channel = input.get_shape().as_list()[-1]
    W = tf.get_variable(name=name + '_Weight', shape=[depth, kernel_size, kernel_size, input_channel, output_channel],
                        initializer=tf.truncated_normal_initializer(0.0, 0.1))
    b = tf.get_variable(name=name + '_bias', shape=[output_channel], initializer=tf.constant_initializer(0.1))
    return tf.add(tf.nn.conv3d(input, W, strides=[1, depth_strides, 1, 1, 1], padding=padding), b)


def conv2d(input, name, kernel_size, output_channel):
    input_channel = input.get_shape().as_list()[-1]
    W = tf.get_variable(name=name + '_Weight', shape=[kernel_size, kernel_size, input_channel, output_channel],
                        initializer=tf.truncated_normal_initializer(0.0, 0.1))
    b = tf.get_variable(name=name + '_bias', shape=[output_channel], initializer=tf.constant_initializer(0.1))
    return tf.add(tf.nn.conv2d(input, W, strides=[1, 1, 1, 1], padding='SAME'), b)


def fc(input, name, output_channel):
    input_channel = input.get_shape().as_list()[-1]
    W = tf.get_variable(name=name + '_Weight', shape=[input_channel, output_channel],
                        initializer=tf.truncated_normal_initializer(0.0, 0.1))
    b = tf.get_variable(name=name + '_bias', shape=[output_channel], initializer=tf.constant_initializer(0.1))
    return tf.matmul(input, W) + b


def batch_norm(input, name, train, decay=0.9, rnn=False):
    shape = input.get_shape()
    if rnn:
        gamma = tf.get_variable(name=name + "_gamma", shape=[shape[-1]], initializer=tf.random_normal_initializer(0.1, 0.001))
    else:
        gamma = tf.get_variable(name = name + "_gamma", shape = [shape[-1]], initializer=tf.random_normal_initializer(1.0, 0.01))
    beta = tf.get_variable(name = name + "_beta", shape = [shape[-1]], initializer=tf.constant_initializer(0.0))

    batch_mean, batch_variance = tf.nn.moments(input, list(range(len(shape) - 1)))

    moving_mean = tf.get_variable(name = name + '_moving_mean', shape = [shape[-1]], initializer=tf.zeros_initializer(), trainable=False)
    moving_variance = tf.get_variable(name = name + '_moving_variance', shape = [shape[-1]], initializer=tf.ones_initializer(), trainable=False)

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

    result = tf.nn.batch_normalization(x = input, mean = mean, variance = variance, offset = beta, scale = gamma, variance_epsilon = 1e-5)
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


def conv2d_Gabor(input, name, theta_num, lambda_num, size, output_channel):
    """
    Theta and Lambda is very important in Gabor filter.
    Theta: [-pi, pi]
    Lambda: [2, image_size/5]
    
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
    output_channel = theta_num * lambda_num
    pi = tf.asin(1.0) * 2.0

    # Theta = tf.get_variable(name=name+"_theta", shape=[theta_num],
    #                         initializer=tf.random_uniform_initializer(minval=-pi, maxval=pi))
    # Lambda = tf.get_variable(name=name+"_lambda", shape=[lambda_num],
    #                          initializer=tf.random_uniform_initializer(minval=2.0, maxval=min(shape[1], shape[2])/5))
    # Theta, Lambda = tf.meshgrid(Theta, Lambda)
    # Theta = tf.tile(input=Expand_dim_up(tf.reshape(Theta, [-1]), 3), multiples=[size, size, input_channel, 1])
    # Lambda = tf.tile(input=Expand_dim_up(tf.reshape(Lambda, [-1]), 3), multiples=[size, size, input_channel, 1])
    # print(Theta.get_shape())

    Theta = tf.get_variable(name=name+"_theta", shape=[1, 1, input_channel, output_channel],
                            initializer=tf.random_uniform_initializer(minval=-pi, maxval=pi))
    Lambda = tf.get_variable(name=name+"_lambda", shape=[1, 1, input_channel, output_channel],
                             initializer=tf.random_uniform_initializer(minval=2.0, maxval=min(shape[1], shape[2])/5))
    Theta = tf.tile(input=Theta, multiples=[size, size, 1, 1])
    Lambda = tf.tile(input=Lambda, multiples=[size, size, 1, 1])
    print(Theta.get_shape())

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
    filter = tf.multiply(tf.exp(-tf.div(tf.add(tf.square(x_), tf.square(y_)), tf.multiply(2.0, tf.square(Sigma)))),
                      tf.cos(2.0 * pi * x_ / Lambda))
    output =tf.nn.conv2d(input, filter, strides=[1, 1, 1, 1], padding='SAME')
    # output =  tf.abs(output)
    return output


batch_size = 50
x = tf.placeholder("float", shape=[batch_size, 32, 32, 3])
y = tf.placeholder("float", shape=[batch_size, 10])
keep_prob = tf.placeholder("float")
BN_train = tf.placeholder("bool", shape=[])

conv1 = conv2d_Gabor(x, 'Gabor1', 8, 8, 7, 64)
# conv1 = conv2d(input=x, name='conv1', kernel_size=3,output_channel=64)
batch1 = batch_norm(input=conv1, name='batch1', train=BN_train)
act1 = tf.nn.relu(batch1)
pool1 = max_pooling_2d(input=act1, width=2, height=2)
#drop1 = tf.nn.dropout(pool1, keep_prob)

conv2 = conv2d(input=pool1, name='conv2', kernel_size=3, output_channel=256)
# conv2 = conv2d_Gabor(pool1, 'Gabor2', 8, 8, 7, 64)
batch2 = batch_norm(input=conv2, name='batch2', train=BN_train)
act2 = tf.nn.relu(batch2)
pool2 = max_pooling_2d(input=act2, width=2, height=2)
drop2 = tf.nn.dropout(pool2, keep_prob)

conv3 = conv2d(input=drop2, name='conv3', kernel_size=3, output_channel=256)
# conv3 = conv2d_Gabor(drop2, 'Gabor3', 8, 8, 7, 64)
batch3 = batch_norm(input=conv3, name='batch3', train=BN_train)
act3 = tf.nn.relu(batch3)
pool3 = max_pooling_2d(input=act3, width=2, height=2)
drop3 = tf.nn.dropout(pool3, keep_prob)

fc1 = fc(input=tf.reshape(drop3, [-1, 4 * 4 * 256]), name='fc1', output_channel=256)
fc1_batch = batch_norm(input=fc1, name='fc_batch1', train=BN_train)
fc1_act = tf.nn.relu(fc1_batch)
drop4 = tf.nn.dropout(fc1_act, keep_prob)

fc2 = fc(input=drop4, name='fc2', output_channel=10)
fc2_batch = batch_norm(input=fc2, name='fc_batch2', train=BN_train)
y_predict = tf.nn.softmax(fc2_batch)

cross_entropy = -tf.reduce_sum(y * tf.log(tf.clip_by_value(y_predict, 1e-10, 1.0)))
train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_predict, 1), tf.argmax(y, 1))

correct_num = tf.reduce_sum(tf.cast(correct_prediction, "float"))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
sess = tf.Session()

DATA = MyDataset('ori_data', batch_size)
train_num = DATA.train_num
test_num = DATA.test_num

sess.run(tf.global_variables_initializer())
variable_name = [v.name for v in tf.trainable_variables()]
for _ in variable_name:
    print(_)

f = open('Gabor_CNN.txt', 'a')
epoch_num = 100
for epoch in range(epoch_num):
    t = time.time()
    train_correct = 0
    for i in range(int(train_num / batch_size)):
        data, label = DATA.train_get_next()
        if len(data) == batch_size:
            num, _ = sess.run([correct_num, train_step], feed_dict={x: data, y: label, BN_train: True, keep_prob: 0.7})
            train_correct += num

    print('epoch:%d ' % epoch)
    print('train accuracy: %f ' % (train_correct / train_num))
    f.write('epoch:%d ' % epoch + '\n')
    f.write('train accuracy: %f ' % (train_correct / train_num) + '\n')

    test_correct = 0
    for i in range(int(DATA.test_num / batch_size)):
        data, label = DATA.test_get_next()
        if len(data) == batch_size:
            num = sess.run(correct_num, feed_dict={x: data, y: label, BN_train: False, keep_prob: 1.0})
            test_correct += num
    print('test accuracy: %f ' % (test_correct / test_num))
    f.write('test accuracy: %f ' % (test_correct / test_num) + '\n')
    print(time.time() - t)