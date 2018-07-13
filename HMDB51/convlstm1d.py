import tensorflow as tf

def conv2d(input, name, kernel_size, input_channel, output_channel):
    # TODO change to 1d
    W = tf.get_variable(name = name + '_Weight', shape = [kernel_size, kernel_size, input_channel, output_channel])
    b = tf.get_variable(name = name + '_bias', shape = [output_channel])
    return tf.add(tf.nn.conv2d(input, W, strides=[1, 1, 1, 1], padding='SAME'), b)

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



    # if output_h:
    #     if pool:
    #         output = max_pooling_2d(input=state[1], height=2, width=2)
    #         return output
    #     else:
    #         return state[1]
    # else:
    #     bn = batch_norm(input=tf.transpose(output, [1, 0, 2, 3, 4]), name=name, train=train)
    #     output = tf.transpose(bn, [1, 0, 2, 3, 4])
    #     if pool:
    #         output = max_pooling_3d(input=output, depth=1, height=2, width=2)
    #     return output


lstm_input = tf.transpose(tf.reshape(pool2, [batch_size, int(depth/5), 8, 10, 128]), [1, 0, 2, 3, 4]) # to fit the time_major
convlstm1 = convlstm_cell(input = lstm_input, name = 'convlstm1', sequence_length=sequence_length, num_filters = 128, kernel_size = [3, 3], train=BN_train, keep_prob=keep_prob)
