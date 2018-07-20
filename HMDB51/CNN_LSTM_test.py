import tensorflow as tf
import numpy as np
import cv2
import os
import time
from My_dataset_class import MyDataset
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def dataset(PATH, batch_size, epoch_num):
    """
    Generate the tf.data.dataset. This function return the iterator and sample size. iterator need to be initial as
        sess.run(iterator.initializer)
    And you can get batch data as 
        sess.run(iterator.get_next())
    Args:
        PATH: 
        batch_size: 
        epoch_num:
        one_hot:

    Returns: [train_iterator, train_num, test_iterator, test_num]

    """
    train_data = []
    train_label = []
    test_data = []
    test_label = []

    train_data_list = os.listdir(PATH + '/numpy/train')
    train_num = len(train_data_list)

    for i in range(train_num):
        train_data.append(PATH + '/numpy/train/' + train_data_list[i] + '/data.npy')
        train_label.append(np.load(PATH + '/numpy/train/' + train_data_list[i] + '/label.npy'))

    test_data_list = os.listdir(PATH + '/numpy/test')
    test_num = len(test_data_list)

    for i in range(test_num):
        test_data.append(PATH + '/numpy/test/' + test_data_list[i] + '/data.npy')
        test_label.append(np.load(PATH + '/numpy/test/' + test_data_list[i] + '/label.npy'))

    print(train_num, test_num)

    train_dataset = tf.data.Dataset.from_tensor_slices((train_data, train_label))
    train_dataset = train_dataset.shuffle(buffer_size=train_num*10).batch(batch_size).repeat(epoch_num)
    train_iterator = train_dataset.make_initializable_iterator()

    test_dataset = tf.data.Dataset.from_tensor_slices((test_data, test_label))
    test_dataset = test_dataset.shuffle(buffer_size=test_num*10).batch(batch_size).repeat(epoch_num)
    test_iterator = test_dataset.make_initializable_iterator()

    return train_iterator, train_num, test_iterator, test_num

def load_prestored_data(PATH, depth, height, width, channel = 3):
    batch_size = len(PATH)
    data = []

    for i in range(batch_size):
        data.append(np.load(PATH))

    return np.array(data)

def max_pooling_3d(input, depth, width, height):
    return tf.nn.max_pool3d(input, ksize=[1, depth, width, height, 1], strides=[1, depth, width, height, 1], padding='SAME')

def max_pooling_2d(input, width, height):
    return tf.nn.max_pool(input, ksize=[1, width, height, 1], strides=[1, width, height, 1], padding='SAME')

def conv3d(input, name, depth, kernel_size, input_channel, output_channel, depth_strides = 1, padding = 'SAME'):
    W = tf.get_variable(name = name + '_Weight', shape = [depth, kernel_size, kernel_size, input_channel, output_channel])
    b = tf.get_variable(name = name + '_bias', shape = [output_channel])
    return tf.add(tf.nn.conv3d(input, W, strides=[1, depth_strides, 1, 1, 1], padding=padding), b)

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

class BasicConvLSTMCell(tf.contrib.rnn.RNNCell):
    def __init__(self, shape, num_filters, kernel_size, name, forget_bias=1.0,
               input_size=None, state_is_tuple=True, activation=tf.nn.tanh, reuse=None):
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

            inp_channel = inputs.get_shape().as_list()[-1]+self._num_filters * 2
            out_channel = self._num_filters * 4
            concat = tf.concat([inputs, h, c], axis=3)

            concat = conv2d(input = concat, name = self._name, kernel_size = self._kernel_size[0], input_channel = inp_channel, output_channel = out_channel)

            i, j, f, o = tf.split(value=concat, num_or_size_splits=4, axis=3)

            new_c = (c * tf.sigmoid(f + self._forget_bias) + tf.sigmoid(i) * self._activation(j))
            new_h = self._activation(new_c) * tf.sigmoid(o)


            if self._state_is_tuple:
                new_state = tf.contrib.rnn.LSTMStateTuple(new_c, new_h)
            else:
                new_state = tf.concat([new_c, new_h], 3)


            return new_h, new_state

def convlstm_cell(input, name, num_filters, kernel_size, train, keep_prob = 1.0, pool = False, output_h = False):
    shape = input.get_shape().as_list() #[time, batch, height, width, channel]
    cell = BasicConvLSTMCell(shape = [shape[2], shape[3]], num_filters = num_filters, kernel_size = kernel_size, name = name)
    cell = tf.contrib.rnn.DropoutWrapper(cell = cell, input_keep_prob = 1.0, output_keep_prob = keep_prob)
    init_state = cell.zero_state(shape[1], dtype=tf.float32)

    output, state = tf.nn.dynamic_rnn(cell, inputs=input, initial_state=init_state, time_major=True)
    # output.get_shape = [time, batch, height, width, channel]
    # state is a tuple

    if output_h:
        if pool:
            output = max_pooling_2d(input=state[1], height=2, width=2)
            return output
        else:
            return state[1]
    else:
        bn = batch_norm(input=tf.transpose(output, [1, 0, 2, 3, 4]), name=name, train=train)
        output = tf.transpose(bn, [1, 0, 2, 3, 4])
        if pool:
            output = max_pooling_3d(input=output, depth=1, height=2, width=2)
        return output


epoch_num = 100
batch_size = 48

depth = 40
height = 32
width = 40

x = tf.placeholder("float", shape = [batch_size, depth, height, width, 3])
y = tf.placeholder("float", shape = [batch_size, 51])
sequence_length = tf.placeholder('int32', shape = [batch_size])
BN_train = tf.placeholder('bool', shape = [])
keep_prob = tf.placeholder("float", shape = [])
learning_rate = tf.placeholder('float', shape = [])

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

lstm_input = tf.transpose(drop2, [1, 0, 2, 3, 4]) # to fit the time_major

print(lstm_input.get_shape())
convlstm1 = convlstm_cell(input = lstm_input, name = 'convlstm1', num_filters = 128, kernel_size = [3, 3], keep_prob = keep_prob, train = BN_train)
convlstm2 = convlstm_cell(input = convlstm1, name = 'convlstm2', num_filters = 256, kernel_size = [3, 3], keep_prob = keep_prob, train = BN_train, pool = True)
convlstm3 = convlstm_cell(input = convlstm2, name = 'convlstm3', num_filters = 256, kernel_size = [3, 3], keep_prob = keep_prob, train = BN_train)

lstm_output = convlstm3[-1, :, :, :, :]
reshape = tf.reshape(lstm_output, [batch_size, 4 * 5 * 256])
fc1 = fc(reshape, name = 'fc1', input_channel = 4 * 5 * 256, output_channel = 256)
fc_batch1 = batch_norm(input = fc1, name = 'fc_batch1', train = BN_train)
fc_act1 = tf.nn.relu(fc_batch1)
fc_drop1 = tf.nn.dropout(fc_act1, keep_prob)

y_predict = tf.nn.softmax(fc(fc_drop1, name = 'fc2', input_channel = 256, output_channel = 51))

cross_entropy = -tf.reduce_sum(y * tf.log(tf.clip_by_value(y_predict, 1e-10, 1.0)))
train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_predict, 1), tf.argmax(y, 1))

correct_num = tf.reduce_sum(tf.cast(correct_prediction, "float"))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
sess = tf.Session()

DATA = MyDataset('data', batch_size)
train_num = DATA.train_num
test_num = DATA.test_num

sess.run(tf.global_variables_initializer())

lr = 1e-3
f = open('CNN_LSTM_res.txt', 'a')
for epoch in range(epoch_num):
    train_correct = 0
    for i in range(int(train_num / batch_size)):
        print(i)
        data, label = DATA.train_get_next()
        if len(data) == batch_size:
            t = time.time()
            data = load_prestored_data(data, depth, height, width)
            print(time.time() - t)
            t = time.time()
            num, _ = sess.run([correct_num, train_step], feed_dict={x: data, y: label, BN_train: True, keep_prob: 0.5, learning_rate: lr})
            print(time.time() - t)
            train_correct += num

    print('epoch:%d ' % epoch)
    print('train accuracy: %f ' % (train_correct / train_num))
    f.write('epoch:%d ' % epoch + '\n')
    f.write('train accuracy: %f ' % (train_correct / train_num) + '\n')

    test_correct = 0
    for i in range(int(DATA.test_num / batch_size)):
        data, label = DATA.test_get_next()
        if len(data) == batch_size:
            data = load_prestored_data(data, depth, height, width)
            num = sess.run(correct_num, feed_dict={x: data, y: label, BN_train: False, keep_prob: 1.0})
            test_correct += num
    print('test accuracy: %f ' % (test_correct / test_num))
    f.write('test accuracy: %f ' % (test_correct / test_num) + '\n')

    if test_correct / test_num > 0.35:
        lr = 1e-4
    else:
        lr = 1e-3