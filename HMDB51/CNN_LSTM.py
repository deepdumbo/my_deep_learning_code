import tensorflow as tf
import numpy as np
import cv2
import os

def prestore(PATH, depth, height, width, channel = 3, one_hot = True):
    try:
        os.mkdir('prestored')
    except:
        pass

    CLASS = os.listdir(PATH)
    CLASS_NUM = len(CLASS)

    for i in range(CLASS_NUM):
        print(i)
        video_list = os.listdir(PATH + '/' + CLASS[i])
        for video in video_list:
            data = np.zeros([depth, height, width, channel])
            if one_hot:
                label = np.zeros([CLASS_NUM])
                label[i] = 1
            else:
                label = i
            length_count = 0

            cap = cv2.VideoCapture(PATH + '/' + CLASS[i] + '/' + video)
            while True:
                ret, frame = cap.read()
                if ret:
                    frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_CUBIC)
                    # cv2.imshow('test', frame)
                    # cv2.waitKey(50)
                    data[length_count, :, :, :] = frame[:, :, :]
                    length_count += 1
                    if length_count == depth:
                        break
                else:
                    break
            video_length = length_count

            # TODO it may need more readable name
            np.save('prestored/' + video.replace('avi', 'npy'), np.array([data, label, video_length]))



def dataset(PATH, batch_size, proportion, one_hot = True):
    CLASS = os.listdir(PATH)
    CLASS_NUM = len(CLASS)

    train_data = np.array([])
    train_label = np.array([])

    test_data = np.array([])
    test_label = np.array([])

    for i in range(CLASS_NUM):
        data_ = os.listdir(PATH + '/' + CLASS[i])
        train_data_ = []
        test_data_ = []
        for j in range(len(data_)):
            if np.random.rand() < proportion:
                test_data_.append(PATH + '/' + CLASS[i] + '/' + data_[j])
            else:
                train_data_.append(PATH + '/' + CLASS[i] + '/' + data_[j])

        test_label_ = np.full([len(test_data_)], i)
        train_label_ = np.full([len(train_data_)], i)

        test_data = np.concatenate((test_data, test_data_), axis=0)
        test_label = np.concatenate((test_label, test_label_), axis=0)
        train_data = np.concatenate((train_data, train_data_), axis=0)
        train_label = np.concatenate((train_label, train_label_), axis=0)
    if one_hot:
        train_label = tf.one_hot(train_label, CLASS_NUM, 1, 0)
        test_label = tf.one_hot(test_label, CLASS_NUM, 1, 0)

    train_num = len(train_data)
    test_num = len(test_data)
    print(train_num, test_num)

    train_dataset = tf.data.Dataset.from_tensor_slices((train_data, train_label))
    train_dataset = train_dataset.shuffle(buffer_size=train_num*2).batch(batch_size).repeat(100)
    train_iterator = train_dataset.make_initializable_iterator()
    train_next_batch = train_iterator.get_next()

    test_dataset = tf.data.Dataset.from_tensor_slices((test_data, test_label))
    test_dataset = test_dataset.shuffle(buffer_size=test_num*2).batch(batch_size).repeat(100)
    test_iterator = test_dataset.make_initializable_iterator()
    test_next_batch = test_iterator.get_next()
    return train_iterator, train_next_batch, train_num, test_iterator, test_next_batch, test_num

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

def convlstm_cell(input, name, sequence_length, num_filters, kernel_size, train, keep_prob, pool = False, output_h = False):
    shape = input.get_shape() #[time, batch, height, width, channel]
    cell = BasicConvLSTMCell(shape = [shape[2], shape[3]], num_filters = num_filters, kernel_size = kernel_size, name = name)
    cell = tf.contrib.rnn.DropoutWrapper(cell = cell, input_keep_prob = 1.0, output_keep_prob = keep_prob)
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
        bn = batch_norm(input=tf.transpose(output, [1, 0, 2, 3, 4]), name=name, train=train)
        output = tf.transpose(bn, [1, 0, 2, 3, 4])
        if pool:
            output = max_pooling_3d(input=output, depth=1, height=2, width=2)
        return output

depth = 100 # 6 * 6 + 4
height = 38
width = 46
batch_size = 16

prestore('hmdb51_org', depth, height, width)


x = tf.placeholder("float", shape = [batch_size, depth, height, width, 3])
y = tf.placeholder("float", shape = [batch_size, 51])
sequence_length = tf.placeholder('int32', shape = [batch_size])
BN_train = tf.placeholder('bool', shape = [])
keep_prob = tf.placeholder("float", shape = [])

x_ = tf.reshape(x, [-1, 5, height, width, 3])
print(x.get_shape())

conv1 = conv3d(input = x_, name = 'conv1', depth = 3, kernel_size = 3, input_channel = 3, output_channel = 64, padding='VALID')
batch1 = batch_norm(input = conv1, name = 'batch1', train = BN_train)
act1 = tf.nn.relu(batch1)
pool1 = max_pooling_3d(input = act1, depth = 1, width = 2, height = 2)
print(pool1.get_shape())

conv2 = conv3d(input = pool1, name = 'conv2', depth = 3, kernel_size = 3, input_channel = 64, output_channel = 128, padding='VALID')
batch2 = batch_norm(input = conv2, name = 'batch2', train = BN_train)
act2 = tf.nn.relu(batch2)
pool2 = max_pooling_3d(input = act2, depth = 1, width = 2, height = 2)

print(pool2.get_shape())

lstm_input = tf.transpose(tf.reshape(pool2, [batch_size, int(depth/5), 8, 10, 128]), [1, 0, 2, 3, 4]) # to fit the time_major
# print(lstm_input.get_shape())
convlstm1 = convlstm_cell(input = lstm_input, name = 'convlstm1', sequence_length=sequence_length, num_filters = 128, kernel_size = [3, 3], train=BN_train, keep_prob=keep_prob)
convlstm2 = convlstm_cell(input = convlstm1, name = 'convlstm2', sequence_length=sequence_length, num_filters = 128, kernel_size = [3, 3], train=BN_train, keep_prob=keep_prob, pool = True)
convlstm3 = convlstm_cell(input = convlstm2, name = 'convlstm3', sequence_length=sequence_length, num_filters = 128, kernel_size = [3, 3], train=BN_train, keep_prob=keep_prob, output_h=True)

reshape = tf.reshape(convlstm3, [-1, 4 * 5 * 128])
fc1 = fc(reshape, name = 'fc1', input_channel = 4 * 5 * 128, output_channel = 512)
fc_act1 = tf.nn.relu(fc1)

y_predict = tf.nn.softmax(fc(fc_act1, name = 'fc2', input_channel = 512, output_channel = 51))

cross_entropy = -tf.reduce_sum(y * tf.log(tf.clip_by_value(y_predict, 1e-10, 1.0)))
train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_predict, 1), tf.argmax(y, 1))

correct_num = tf.reduce_sum(tf.cast(correct_prediction, "float"))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
sess = tf.Session()

train_iterator, train_next_batch, train_num, test_iterator, test_next_batch, test_num = dataset('hmdb51_org', batch_size, proportion=0.2)
sess.run(tf.global_variables_initializer())
sess.run(train_iterator.initializer)
sess.run(test_iterator.initializer)

for epoch in range(0):
    train_correct = 0
    print(epoch, 'train:')
    for i in range(int(train_num / batch_size)):
        data, label = sess.run(train_next_batch)
        if len(data) == batch_size:
            data, length = load_video(data, depth, height, width)
            length = np.array(length/5).astype(int)

            num, _ = sess.run([correct_num, train_step], feed_dict={x: data, y: label, sequence_length: length, BN_train: True, keep_prob: 0.5})
            train_correct += num
        if i%20==19:
            print((i+1)*batch_size, train_correct/(i+1)/batch_size)

    test_correct = 0
    print(epoch, 'test:')
    for i in range(int(test_num / batch_size)):
        data, label = sess.run(test_next_batch)
        if len(data) == batch_size:
            data, length = load_video(data, depth, height, width)
            length = np.array(length/5).astype(int)

            num = sess.run(correct_num, feed_dict={x: data, y: label, sequence_length: length, BN_train: False, keep_prob: 1.0})
            test_correct += num
        if i%20==19:
            print((i+1)*batch_size, test_correct/(i+1)/batch_size)