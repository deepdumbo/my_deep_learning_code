import numpy as np
import tensorflow as tf
import os
import cv2

def load_img(filenames):
    img = []
    for filename in filenames:
        img.append(cv2.imread(filename.decode()))
    return img

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

    dataset = tf.data.Dataset.from_tensor_slices((data, label))
    dataset = dataset.shuffle(buffer_size=total_num*2).batch(batch_size).repeat(10)
    iterator = dataset.make_initializable_iterator()
    next_batch = iterator.get_next()
    return iterator, next_batch

def conv2d(input, name, kernel_size, input_channel, output_channel):
    W = tf.get_variable(name=name + '_Weight', shape=[kernel_size, kernel_size, input_channel, output_channel])
    b = tf.get_variable(name=name + '_bias', shape=[output_channel])
    return tf.add(tf.nn.conv2d(input, W, strides=[1, 1, 1, 1], padding='SAME'), b)

def max_pooling_2d(input, width, height):
    return tf.nn.max_pool(input, ksize=[1, width, height, 1], strides=[1, width, height, 1], padding='SAME')

def fc(input, name, input_channel, output_channel):
    W = tf.get_variable(name=name + '_Weight', shape=[input_channel, output_channel])
    b = tf.get_variable(name=name + '_bias', shape=[output_channel])
    return tf.matmul(input, W) + b

batch_size = 100
x = tf.placeholder("float", shape=[batch_size, 32, 32, 3])
y = tf.placeholder("float", shape=[batch_size, 10])

conv1 = conv2d(input=x, name='conv1', kernel_size=3, input_channel=3, output_channel=32)
act1 = tf.nn.relu(conv1)
pool1 = max_pooling_2d(input=act1, width=2, height=2)

conv2 = conv2d(input=pool1, name='conv2', kernel_size=3, input_channel=32, output_channel=32)
act2 = tf.nn.relu(conv2)
pool2 = max_pooling_2d(input=act2, width=2, height=2)

conv3 = conv2d(input=pool2, name='conv3', kernel_size=3, input_channel=32, output_channel=32)
act3 = tf.nn.relu(conv3)
pool3 = max_pooling_2d(input=act3, width=2, height=2)

fc1 = fc(input=tf.reshape(pool3, [-1, 4 * 4 * 32]), name='fc1', input_channel=4 * 4 * 32, output_channel=64)
fc1_act = tf.nn.relu(fc1)

fc2 = fc(input=fc1_act, name='fc2', input_channel=64, output_channel=10)
y_predict = tf.nn.softmax(fc2)
cross_entropy = -tf.reduce_sum(y * tf.log(tf.clip_by_value(y_predict, 1e-10, 1.0)))
train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_predict, 1), tf.argmax(y, 1))

correct_num = tf.reduce_sum(tf.cast(correct_prediction, "float"))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

train_iterator, train_next_batch = dataset('train', batch_size)
test_iterator, test_next_batch = dataset('test', batch_size)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
sess.run(train_iterator.initializer)
sess.run(test_iterator.initializer)

for epoch in range(10):
    train_correct = 0
    for i in range(int(50000 / batch_size)):
        data, label = sess.run(train_next_batch)
        data = load_img(data)

        num, _ = sess.run([correct_num, train_step], feed_dict={x: data, y: label})
        train_correct += num
        if i % 50 == 49:
            print(i*batch_size, train_correct/i/batch_size)


