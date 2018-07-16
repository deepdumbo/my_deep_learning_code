# -*- coding:utf-8 -*-
import numpy as np
import cv2
import os
import tensorflow as tf
import random

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def max_pooling_3d(input, depth, width, height):
    return tf.nn.max_pool3d(input, ksize=[1, depth, width, height, 1], strides=[1, depth, width, height, 1], padding='SAME')


def max_pooling_2d(input, width, height):
    return tf.nn.max_pool(input, ksize=[1, width, height, 1], strides=[1, width, height, 1], padding='SAME')

def conv3d(input, name, depth, kernel_size, input_channel, output_channel, depth_strides = 1):
    W = tf.get_variable(name = name + '_Weight', shape = [depth, kernel_size, kernel_size, input_channel, output_channel])
    b = tf.get_variable(name = name + '_bias', shape = [output_channel])
    return tf.add(tf.nn.conv3d(input, W, strides=[1, depth_strides, 1, 1, 1], padding='SAME'), b)

def conv2d(input, name, kernel_size, input_channel, output_channel):
    W = tf.get_variable(name = name + '_Weight', shape = [kernel_size, kernel_size, input_channel, output_channel])
    b = tf.get_variable(name = name + '_bias', shape = [output_channel])
    return tf.add(tf.nn.conv2d(input, W, strides=[1, 1, 1, 1], padding='SAME'), b)

def conv2d_to_1_mul_1(input, name, input_channel, output_channel):
    shape = input.get_shape()
    height = shape[1]
    width = shape[2]
    W = tf.get_variable(name = name + '_Weight', shape = [height, width, input_channel, output_channel])
    b = tf.get_variable(name = name + '_bias', shape = [output_channel])
    return tf.add(tf.nn.conv2d(input, W, strides=[1, 1, 1, 1], padding='VALID'), b)

def fc(input, name, input_channel, output_channel):
    W = tf.get_variable(name=name + '_Weight', shape=[input_channel, output_channel])
    b = tf.get_variable(name=name + '_bias', shape=[output_channel])
    return tf.matmul(input, W) + b

def batch_norm(input, name, train, decay = 0.9):
    shape = input.get_shape()
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

def lstm_cell(hidden_size):
    cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_size, forget_bias=1.0, state_is_tuple=True)
    #add dropout layer
    cell = tf.contrib.rnn.DropoutWrapper(cell=cell, input_keep_prob=1.0, output_keep_prob=keep_prob)
    return cell


class BasicConvLSTMCell(tf.contrib.rnn.RNNCell):
    def __init__(self, shape, num_filters, kernel_size, name, forget_bias=1.0,
               input_size=None, state_is_tuple=True, activation=tf.nn.softsign, reuse=None):
        self._shape = shape
        self._num_filters = num_filters
        self._kernel_size = kernel_size
        self._size = tf.TensorShape(shape+[self._num_filters])

        self._forget_bias = forget_bias
        self._state_is_tuple = state_is_tuple
        self._activation = activation
        self._name = name


        self._reuse = reuse

    @property
    def state_size(self):
        return (tf.contrib.rnn.LSTMStateTuple(self._size, self._size)
                if self._state_is_tuple else 2 * self._num_units)

    @property
    def output_size(self):
        return self._size

    def __call__(self, inputs, state, scope=None):
        # we suppose inputs to be [time, batch_size, row, col, channel]
        with tf.variable_scope(scope or "basic_convlstm_cell", reuse=self._reuse):
            if self._state_is_tuple:
                c, h = state
            else:
                c, h = tf.split(value=state, num_or_size_splits=2, axis=3)

            inp_channel = inputs.get_shape().as_list()[-1]+self._num_filters
            out_channel = self._num_filters * 4
            concat = tf.concat([inputs, h], axis=3)

            concat = conv2d(input = concat, name = self._name, kernel_size = self._kernel_size[0], input_channel = inp_channel, output_channel = out_channel)

            i, j, f, o = tf.split(value=concat, num_or_size_splits=4, axis=3)

            new_c = (c * tf.sigmoid(f + self._forget_bias) + tf.sigmoid(i) * self._activation(j))
            new_h = self._activation(new_c) * tf.sigmoid(o)


            if self._state_is_tuple:
                new_state = tf.contrib.rnn.LSTMStateTuple(new_c, new_h)
            else:
                new_state = tf.concat([new_c, new_h], 3)


            return new_h, new_state





def convlstm_cell(input, name, num_filters, kernel_size, keep_prob, batch_size, train, pool = False):
    shape = input.get_shape() #[time, batch, height, width, channel]
    cell = BasicConvLSTMCell(shape = [shape[2], shape[3]], num_filters = num_filters, kernel_size = kernel_size, name = name)
    cell = tf.contrib.rnn.DropoutWrapper(cell = cell, input_keep_prob = 1.0, output_keep_prob = keep_prob)
    init_state = cell.zero_state(batch_size, dtype=tf.float32)

    output, state = tf.nn.dynamic_rnn(cell, inputs=input, initial_state=init_state, time_major=True)
    # output.get_shape = [time, batch, height, width, channel]
    # state is a tuple


    bn = batch_norm(input = tf.transpose(output, [1, 0, 2, 3, 4]), name = name, train = train)
    output = tf.transpose(bn, [1, 0, 2, 3, 4])
    if pool:
        output = max_pooling_3d(input = output, depth = 1, height = 2, width = 2)

    return output


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



print('read data...')
PATH = 'data/numpy/'
train_data = np.load(PATH + "train_data.npy")
train_label = np.load(PATH + "train_label.npy")
test_data = np.load(PATH + "test_data.npy")
test_label = np.load(PATH + "test_label.npy")
print('finish')

depth = 40 # 6 * 6 + 4
height = 32
width = 40
batch_size = tf.placeholder('int32', shape = [])
x = tf.placeholder("float", shape = [None, depth, height, width, 3])
y = tf.placeholder("float", shape = [None, 51])
lr = tf.placeholder("float", shape = [])
keep_prob = tf.placeholder("float", shape = [])
BN_train = tf.placeholder("bool", shape=[])

"""
10000010000010
10101010101010
11111111111111
"""



conv1 = conv3d(input = x, name = 'conv1', depth = 3, kernel_size = 3, input_channel = 3, output_channel = 64, depth_strides = 1)
batch1 = batch_norm(input = conv1, name = 'batch1', train = BN_train)
act1 = tf.nn.relu(batch1)
pool1 = max_pooling_3d(input = act1, depth = 1, width = 2, height = 2)
drop1 = tf.nn.dropout(pool1, keep_prob)

conv2 = conv3d(input = drop1, name = 'conv2', depth = 3, kernel_size = 3, input_channel = 64, output_channel = 128, depth_strides = 1)
batch2 = batch_norm(input = conv2, name = 'batch2', train = BN_train)
act2 = tf.nn.relu(batch2)
pool2 = max_pooling_3d(input = act2, depth = 5, width = 2, height = 2)
drop2 = tf.nn.dropout(pool2, keep_prob)

print(conv1.get_shape())
print(drop2.get_shape())
lstm_input = tf.transpose(drop2, [1, 0, 2, 3, 4]) # to fit the time_major

convlstm1 = convlstm_cell(input = lstm_input, name = 'convlstm1', num_filters = 128, kernel_size = [3, 3], keep_prob = keep_prob, batch_size = batch_size, train = BN_train)
convlstm2 = convlstm_cell(input = convlstm1, name = 'convlstm2', num_filters = 256, kernel_size = [3, 3], keep_prob = keep_prob, batch_size = batch_size, train = BN_train, pool = True)
convlstm3 = convlstm_cell(input = convlstm2, name = 'convlstm3', num_filters = 256, kernel_size = [3, 3], keep_prob = keep_prob, batch_size = batch_size, train = BN_train)

lstm_output = convlstm3[-1, :, :, :, :]
reshape = tf.reshape(lstm_output, [batch_size, 4 * 5 * 256])
fc1 = fc(reshape, name = 'fc1', input_channel = 4 * 5 * 256, output_channel = 256)
fc_batch1 = batch_norm(input = fc1, name = 'fc_batch1', train = BN_train)
fc_act1 = tf.nn.relu(fc_batch1)
fc_drop1 = tf.nn.dropout(fc_act1, keep_prob)

y_predict = tf.nn.softmax(fc(fc_drop1, name = 'fc2', input_channel = 256, output_channel = 51))


cross_entropy = -tf.reduce_sum(y * tf.log(tf.clip_by_value(y_predict, 1e-10, 1.0)))
train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)
#train_step = tf.train.GradientDescentOptimizer(lr).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_predict, 1), tf.argmax(y, 1))

correct_num = tf.reduce_sum(tf.cast(correct_prediction, "float"))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))




config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
sess = tf.Session()
sess.run(tf.global_variables_initializer())


train_batch_size = 25
test_batch_size = 25
train_batch_image = np.zeros([train_batch_size, depth, height, width, 3])
train_batch_label = np.zeros([train_batch_size, 11])
test_batch_image = np.zeros([test_batch_size, depth, height, width, 3])
test_batch_label = np.zeros([test_batch_size, 11])
learning_rate = 1e-3
f = open('5_pool.txt', 'a')
for epoch in range(200):
    train_data, train_label, test_data, test_label = data_shullfe(train_data, train_label, test_data, test_label)

    train_correct = 0
    train_num = 5072
    for j in range(int(train_num / train_batch_size)):
        train_batch_image = train_data[j*train_batch_size : (j+1)*train_batch_size, :, :, :, :]
        train_batch_label = train_label[j*train_batch_size : (j+1)*train_batch_size, :]

        num, _ = sess.run([correct_num, train_step], feed_dict={batch_size: train_batch_size, x: train_batch_image, y: train_batch_label, keep_prob: 0.5, BN_train: True, lr: learning_rate})
        train_correct += num
    print('epoch:%d ' % epoch)
    print('train accuracy: %f ' % (train_correct / train_num))
    f.write('epoch:%d ' % epoch + '\n')
    f.write('train accuracy: %f ' % (train_correct / train_num) + '\n')

    test_correct = 0
    test_num = 1302
    for j in range(int(test_num / test_batch_size)):
        test_batch_image = test_data[j*test_batch_size : (j+1)*test_batch_size, :, :, :, :]
        test_batch_label = test_label[j*test_batch_size : (j+1)*test_batch_size, :]

        num =  sess.run(correct_num, feed_dict={batch_size: test_batch_size, x: test_batch_image, y: test_batch_label, keep_prob: 1.0, BN_train: False})
        test_correct += num
    print('test accuracy: %f ' % (test_correct / test_num))
    f.write('test accuracy: %f ' % (test_correct / test_num) + '\n')


    if train_correct / train_num > 0.43:
        learning_rate = 1e-4
    else:
        learning_rate = 1e-3



