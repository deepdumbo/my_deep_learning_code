import tensorflow as tf
import numpy as np
import cv2
import os

def read_video(PATH):
    CLASS = os.listdir(PATH)
    CLASS_NUM = len(CLASS)

    max_frame = 0
    for i in range(CLASS_NUM):
        video_list = os.listdir(PATH + '/' + CLASS[i])
        for video in video_list:
            cap = cv2.VideoCapture(PATH + '/' + CLASS[i] + '/' + video)
            cnt = 0
            while True:
                ret, frame = cap.read()
                if ret:
                    cnt += 1
                else:
                    print(PATH + '/' + CLASS[i] + '/' + video, cnt)
                    max_frame = max(max_frame, cnt)
                    break
    print(max_frame)

def dataset(PATH, batch_size, one_hot = True):
    CLASS = os.listdir(PATH)
    CLASS_NUM = len(CLASS)

    data = np.array([])
    label = np.array([])
    for i in range(CLASS_NUM):
        data_ = os.listdir(PATH + '/' + CLASS[i])
        for j in range(len(data_)):
            data_[j] = PATH + '/' + CLASS[i] + '/' + data_[j]
        label_ = np.full([len(data_)], i)

        data = np.concatenate((data, data_), axis=0)
        label = np.concatenate((label, label_), axis=0)
    if one_hot:
        label = tf.one_hot(label, CLASS_NUM, 1, 0)
    total_num = len(data)
    print(total_num)

    dataset = tf.data.Dataset.from_tensor_slices((data, label))
    dataset = dataset.shuffle(buffer_size=total_num*2).batch(batch_size).repeat(10)
    iterator = dataset.make_initializable_iterator()
    next_batch = iterator.get_next()
    return iterator, next_batch

def load_video(filenames, depth, height, width, channel = 3):
    batch_size = len(filenames)

    video = np.zeros([batch_size, depth, height, width, channel])
    video_length = np.zeros([batch_size])

    for i in range(batch_size):
        cap = cv2.VideoCapture(filenames[i].decode())
        length_count = 0
        while True:
            ret, frame = cap.read()
            # print(np.shape(frame))
            if ret:
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_CUBIC)
                # cv2.imshow('test', frame)
                # cv2.waitKey(50)
                video[i, length_count, :, :, :] = frame[:, :, :]
                length_count += 1
                if length_count == depth:
                    break
            else:
                break
        video_length[i] = length_count

    return video, video_length

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
                c, h = array_ops.split(value=state, num_or_size_splits=2, axis=3)

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

def convlstm_cell(input, name, sequence_length, num_filters, kernel_size, pool = False, output_h = False):
    shape = input.get_shape() #[time, batch, height, width, channel]
    cell = BasicConvLSTMCell(shape = [shape[2], shape[3]], num_filters = num_filters, kernel_size = kernel_size, name = name)
    # cell = tf.contrib.rnn.DropoutWrapper(cell = cell, input_keep_prob = 1.0, output_keep_prob = keep_prob)
    init_state = cell.zero_state(batch_size, dtype=tf.float32)

    output, state = tf.nn.dynamic_rnn(cell, inputs=input, sequence_length=sequence_length, initial_state=init_state, time_major=True)
    # output.get_shape = [time, batch, height, width, channel]
    # state is a tuple
    if output_h:
        if pool:
            output = max_pooling_2d(input=state[1], height=2, width=2)
            return output
        else:
            return state[1]
    else:
        if pool:
            output = max_pooling_3d(input=output, depth=1, height=2, width=2)
        return output

depth = 100 # 6 * 6 + 4
height = 32
width = 40
batch_size = 16

x = tf.placeholder("float", shape = [batch_size, depth, height, width, 3])
y = tf.placeholder("float", shape = [batch_size, 51])
sequence_length = tf.placeholder('int32', shape = [batch_size])

lstm_input = tf.transpose(x, [1, 0, 2, 3, 4]) # to fit the time_major

convlstm1 = convlstm_cell(input = lstm_input, name = 'convlstm1', sequence_length=sequence_length, num_filters = 32, kernel_size = [3, 3], pool = True)
convlstm2 = convlstm_cell(input = convlstm1, name = 'convlstm2', sequence_length=sequence_length, num_filters = 32, kernel_size = [3, 3], pool = True)
convlstm3 = convlstm_cell(input = convlstm2, name = 'convlstm3', sequence_length=sequence_length, num_filters = 32, kernel_size = [3, 3], pool = True, output_h=True)

reshape = tf.reshape(convlstm3, [batch_size, 4 * 5 * 32])
fc1 = fc(reshape, name = 'fc1', input_channel = 4 * 5 * 32, output_channel = 128)
fc_act1 = tf.nn.relu(fc1)

y_predict = tf.nn.softmax(fc(fc_act1, name = 'fc2', input_channel = 128, output_channel = 51))

cross_entropy = -tf.reduce_sum(y * tf.log(tf.clip_by_value(y_predict, 1e-10, 1.0)))
train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_predict, 1), tf.argmax(y, 1))

correct_num = tf.reduce_sum(tf.cast(correct_prediction, "float"))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

sess = tf.Session()

train_iterator, train_next_batch = dataset('hmdb51_org', batch_size)
sess.run(tf.global_variables_initializer())
sess.run(train_iterator.initializer)

for epoch in range(10):
    train_correct = 0
    for i in range(int(6766 / batch_size)):
        data, label = sess.run(train_next_batch)
        data, length = load_video(data, depth, height, width)

        num, _ = sess.run([correct_num, train_step], feed_dict={x: data, y: label, sequence_length: length})
        train_correct += num
        print((i+1)*batch_size, train_correct/(i+1)/batch_size)
