# -*- coding:utf-8 -*-
import pickle as p
import numpy as np
import tensorflow as tf
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def max_pooling_3d(input, depth, width, height):
    return tf.nn.max_pool3d(input, ksize=[1, depth, width, height, 1], strides=[1, depth, width, height, 1],
                            padding='SAME')


def max_pooling_2d(input, width, height):
    return tf.nn.max_pool(input, ksize=[1, width, height, 1], strides=[1, width, height, 1], padding='SAME')


def conv3d(input, name, depth, kernel_size, input_channel, output_channel, depth_strides=1):
    W = tf.get_variable(name=name + '_Weight', shape=[depth, kernel_size, kernel_size, input_channel, output_channel])
    b = tf.get_variable(name=name + '_bias', shape=[output_channel])
    return tf.add(tf.nn.conv3d(input, W, strides=[1, depth_strides, 1, 1, 1], padding='SAME'), b)


def conv2d(input, name, kernel_size, input_channel, output_channel, use_bias = True):
    W = tf.get_variable(name=name + '_Weight', shape=[kernel_size, kernel_size, input_channel, output_channel])
    if use_bias:
        b = tf.get_variable(name=name + '_bias', shape=[output_channel])
        return tf.add(tf.nn.conv2d(input, W, strides=[1, 1, 1, 1], padding='SAME'), b)
    else:
        return tf.nn.conv2d(input, W, strides=[1, 1, 1, 1], padding='SAME')


def conv2d_to_1_mul_1(input, name, input_channel, output_channel):
    shape = input.get_shape()
    height = shape[1]
    width = shape[2]
    W = tf.get_variable(name=name + '_Weight', shape=[height, width, input_channel, output_channel])
    b = tf.get_variable(name=name + '_bias', shape=[output_channel])
    return tf.add(tf.nn.conv2d(input, W, strides=[1, 1, 1, 1], padding='VALID'), b)


def fc(input, name, input_channel, output_channel, use_bias = True):
    W = tf.get_variable(name=name + '_Weight', shape=[input_channel, output_channel])
    if use_bias:
        b = tf.get_variable(name=name + '_bias', shape=[output_channel])
        return tf.matmul(input, W) + b
    else:
        return tf.matmul(input, W)


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


def grid_constant(height, width):
    grid = np.zeros([height, width, 2])
    for i in range(height):
        for j in range(width):
            grid[i][j][0] = j - width / 2
            grid[i][j][1] = i - height / 2
    return tf.constant(grid)


def Expand_dim_down(input, num):
    res = input
    rank = tf.shape(input).get_shape().as_list()[0]
    for i in range(rank, rank + num):
        res = tf.expand_dims(res, i)
    return res


def Expand_dim_up(input, num):
    res = input
    for i in range(num):
        res = tf.expand_dims(res, 0)
    return res


def Gabor_filter(Theta, Lambda, size, in_channel, out_channel):
    # TODO use tf.meshgird instead of code as followed
    coordinate_begin = - (size - 1) / 2.0
    coordinate_end = - coordinate_begin
    tmp = tf.linspace(coordinate_begin, coordinate_end, size)
    tmp = Expand_dim_down(tmp, 3)

    # why i can't just use in_channel as tensor and size as np.int?
    # that is stupid error!
    # and the error information is very confusing!
    # i just do nothing for a day
    # 2018.4.8
    x = tf.tile(tmp, [size, 1, in_channel, out_channel])
    x = tf.reshape(x, [size, size, in_channel, out_channel])
    y = tf.tile(tmp, [1, size, in_channel, out_channel])

    Theta = tf.reshape(tf.tile(tf.expand_dims(Theta, 0), [4, 1]), [-1])
    Lambda = tf.reshape(tf.tile(tf.expand_dims(Lambda, 1), [1, 4]), [-1])

    Theta = tf.tile(Expand_dim_up(Theta, 3), [size, size, in_channel, 1])
    Lambda = tf.tile(Expand_dim_up(Lambda, 3), [size, size, in_channel, 1])

    Sigma = tf.multiply(0.56, Lambda)

    x_ = tf.add(tf.multiply(x, tf.cos(Theta)), tf.multiply(y, tf.sin(Theta)))
    y_ = tf.add(-tf.multiply(x, tf.sin(Theta)), tf.multiply(y, tf.cos(Theta)))
    pi = tf.asin(1.0) * 2

    res = tf.multiply(tf.exp(-tf.div(tf.add(tf.square(x_), tf.square(y_)), tf.multiply(2.0, tf.square(Sigma)))),
                      tf.cos(2.0 * pi * x_ / Lambda))
    # print tf.shape(res).eval()
    return res


def Gabor_conv(input, Theta, Lambda, name, kernel_size, in_channel, out_channel):
    input_shape = input.get_shape()
    batch_size = input_shape[0]

    res = tf.get_variable(name=name, shape=[input_shape[0], input_shape[1], input_shape[2], out_channel],
                          initializer=tf.constant_initializer(0.0), trainable=False)

    # different Gabor conv for each image of batch
    for i in range(batch_size):
        img = Expand_dim_up(input=input[i], num=1)
        Theta_ = Theta[i]
        Lambda_ = Lambda[i]

        Gabor = Gabor_filter(Theta=Theta_, Lambda=Lambda_, size=kernel_size, in_channel=in_channel,
                             out_channel=out_channel)

        img_Gabor_conv = tf.nn.conv2d(img, Gabor, strides=[1, 1, 1, 1], padding='SAME')
        tmp = tf.identity(img_Gabor_conv)

        indices = tf.constant([[i]])
        res = tf.scatter_nd_update(ref=res, indices=indices, updates=tmp)

    return res


print("read data...")
train_data, train_label, test_data, test_label = load_CIFAR("ori_data")
print("finish")

batch_size = 100
x = tf.placeholder("float", shape=[batch_size, 32, 32, 3])
y = tf.placeholder("float", shape=[batch_size, 10])
keep_prob = tf.placeholder("float", shape=[])
BN_train = tf.placeholder("bool", shape=[])

pi = tf.asin(1.0) * 2.0
Theta = tf.linspace(0.0, pi * 1.5, 4)
Lambda = tf.linspace(2.0, 6.0, 4)
Gabor = Gabor_filter(Theta, Lambda, 17, 3, 16)
Gabor_conv_ = tf.nn.conv2d(x, Gabor, strides=[1, 1, 1, 1], padding='SAME')

G_conv1 = conv2d(input=Gabor_conv_, name='G_conv1', kernel_size=3, input_channel=16, output_channel=64, use_bias=False)
G_batch1 = batch_norm(input=G_conv1, name='G_batch1', train=BN_train)
G_act1 = tf.nn.relu(G_batch1)
G_pool1 = max_pooling_2d(input=G_act1, width=2, height=2)

conv1 = conv2d(input=x, name='conv1', kernel_size=3, input_channel=3, output_channel=64)
conv1_ = conv2d(input=conv1, name='conv1_', kernel_size=3, input_channel=64, output_channel=64, use_bias=False)
batch1 = batch_norm(input=conv1_, name='batch1', train=BN_train)
act1 = tf.nn.relu(batch1)
pool1 = max_pooling_2d(input=act1, width=2, height=2)

concat = tf.concat([pool1, G_pool1], axis = -1)
#drop1 = tf.nn.dropout(concat, keep_prob)

conv2 = conv2d(input=concat, name='conv2', kernel_size=3, input_channel=128, output_channel=256)
batch2 = batch_norm(input=conv2, name='batch2', train=BN_train)
act2 = tf.nn.relu(batch2)
pool2 = max_pooling_2d(input=act2, width=2, height=2)
drop2 = tf.nn.dropout(pool2, keep_prob)

conv3 = conv2d(input=drop2, name='conv3', kernel_size=3, input_channel=256, output_channel=256)
batch3 = batch_norm(input=conv3, name='batch3', train=BN_train)
act3 = tf.nn.relu(batch3)
pool3 = max_pooling_2d(input=act3, width=2, height=2)
drop3 = tf.nn.dropout(pool3, keep_prob)

fc1 = fc(input=tf.reshape(drop3, [-1, 4 * 4 * 256]), name='fc1', input_channel=4 * 4 * 256, output_channel=256)
fc1_batch = batch_norm(input=fc1, name='fc_batch', train=BN_train)
fc1_act = tf.nn.relu(fc1_batch)
#drop4 = tf.nn.dropout(fc1_act, keep_prob)

fc2 = fc(input=fc1_act, name='fc2', input_channel=256, output_channel=10)
y_predict = tf.nn.softmax(fc2)
cross_entropy = -tf.reduce_sum(y * tf.log(tf.clip_by_value(y_predict, 1e-10, 1.0)))
train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_predict, 1), tf.argmax(y, 1))

correct_num = tf.reduce_sum(tf.cast(correct_prediction, "float"))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

batch_image = np.zeros([batch_size, 32, 32, 3])
batch_label = np.zeros([batch_size, 10])

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
sess = tf.Session()
sess.run(tf.global_variables_initializer())

for epoch in range(200):
    train_data, train_label, test_data, test_label = data_shullfe(train_data, train_label, test_data, test_label)

    train_correct = 0
    for j in range(int(50000 / batch_size)):
        batch_image = train_data[j * batch_size: (j + 1) * batch_size, :, :, :]
        batch_label = train_label[j * batch_size: (j + 1) * batch_size, :]

        _, num = sess.run([train_step, correct_num], feed_dict={x: batch_image, y: batch_label, keep_prob: 0.5, BN_train: True})
        train_correct += num
    print('epoch:%d ' % epoch)
    print('train accuracy: %f ' % (train_correct / 50000))

    test_correct = 0
    for j in range(int(10000 / batch_size)):
        batch_image = test_data[j * batch_size: (j + 1) * batch_size, :, :, :]
        batch_label = test_label[j * batch_size: (j + 1) * batch_size, :]

        num = sess.run(correct_num, feed_dict={x: batch_image, y: batch_label, keep_prob: 0.5, BN_train: True})
        test_correct += num
    print('test accuracy: %f ' % (test_correct / 10000))





