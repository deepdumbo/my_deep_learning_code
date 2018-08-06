# -*- coding:utf-8 -*-
import pickle as p
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as plimg
import matplotlib.cm as cm
#from PIL import Image
import tensorflow as tf

def load_CIFAR_batch(filename):
    """ load single batch of cifar """
    with open(filename, 'rb')as f:
        datadict = p.load(f)
        X = datadict['data']
        Y = datadict['labels']
        X = X.reshape(10000, 3, 32, 32)
        Y = np.array(Y)
        return X, Y

def load_CIFAR(Foldername):
    train_data = np.zeros([50000,32,32,3])
    train_label = np.zeros([50000,10])
    for sample in range(5):
        X,Y = load_CIFAR_batch(Foldername+"/data_batch_"+str(sample+1))

        for i in range(3):
            train_data[10000*sample:10000*(sample+1),:,:,i] = X[:,i,:,:]
        for i in range(10000):
            train_label[i+10000*sample][Y[i]] = 1
    
    test_data = np.zeros([10000,32,32,3])
    test_label = np.zeros([10000,10])
    X,Y = load_CIFAR_batch(Foldername+"/test_batch")
    for i in range(3):
        test_data[0:10000,:,:,i] = X[:,i,:,:]
    for i in range(10000):
        test_label[i][Y[i]] = 1
    
    return train_data, train_label, test_data, test_label


def weight_variable(shape):
  # initial = tf.truncated_normal(shape, stddev=0.1)
  # return tf.Variable(initial)
  global cnt
  cnt = cnt  + 1
  return tf.get_variable(shape=shape, name='Weight' + str(cnt))


def bias_variable(shape):
  # initial = tf.constant(0.1, shape=shape)
  # return tf.Variable(initial)
  global cnt
  cnt = cnt  + 1
  return tf.get_variable(shape=shape, name='Bias' + str(cnt))

def convolution_2d(x, W, b):
  return tf.add(tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME'), b)

def max_pooling(x, width, height):
  return tf.nn.max_pool(x, ksize=[1, width, height, 1], strides=[1, width, height, 1], padding='SAME')

def grid_constant(height, width):
    grid = np.zeros([height, width, 2])
    for i in range(height):
        for j in range(width):
            grid[i][j][0] = j - width / 2
            grid[i][j][1] = i - height / 2
    return tf.constant(grid)


def Expand_dim_down(input, num):
    res = input
    for i in range(1, num + 1):
        res = tf.expand_dims(res, i)
    return res

def Expand_dim_up(input, num):
    res = input
    for i in range(num):
        res = tf.expand_dims(res, 0)
    return res

def Gabor_filter(Theta, Lambda, Fai, Sigma, Gamma, size, in_channel, out_channel):

    Theta_num = tf.shape(Theta)[0]
    Lambda_num = tf.shape(Lambda)[0]
    print Theta_num

    coordinate_begin = - (size - 1) / 2.0
    coordinate_end = - coordinate_begin
    tmp = tf.linspace(coordinate_begin, coordinate_end, size)
    tmp = Expand_dim_down(tmp, 3)
    x = tf.tile(tmp, [size, 1, in_channel, out_channel])
    x = tf.reshape(x, [size, size, in_channel, out_channel])
    y = tf.tile(tmp, [1, size, in_channel, out_channel])

    Theta = tf.reshape(tf.tile(tf.expand_dims(Theta, 0), [Lambda_num, 1]), [-1])
    Lambda = tf.reshape(tf.tile(tf.expand_dims(Lambda, 1), [1, Theta_num]), [-1])

    Theta = tf.tile(Expand_dim_up(Theta, 3), [size, size, in_channel, 1])
    Lambda = tf.tile(Expand_dim_up(Lambda, 3), [size, size, in_channel, 1])

    Sigma = tf.multiply(0.56, Lambda)

    x_ = tf.add(tf.multiply(x, tf.cos(Theta)), tf.multiply(y, tf.sin(Theta)))
    y_ = tf.add(-tf.multiply(x, tf.sin(Theta)), tf.multiply(y, tf.cos(Theta)))
    pi = tf.asin(1.0) * 2

    res = tf.multiply(tf.exp(-tf.div(tf.add(tf.square(x_), tf.square(y_)), tf.multiply(2.0, tf.square(Sigma)))), tf.cos(2.0 * pi * x_ / Lambda))
    print tf.shape(res).eval()
    return res






    #print Theta.eval(), tf.shape(Theta).eval()
    #print Lambda.eval(), tf.shape(Lambda).eval()





    #for i in range(4)
    #print x.eval(),tf.shape(x).eval()
    #print y.eval(),tf.shape(y).eval()


print "read data..."

train_data, train_label, test_data, test_label = load_CIFAR("dataset")
print "finish"


sess = tf.InteractiveSession()


batch_size = 50
x = tf.placeholder("float", shape=[batch_size, 32, 32, 3])
y = tf.placeholder("float", shape=[batch_size, 10])
rate = tf.placeholder("float")
keep_prob = tf.placeholder("float")
BN_train = tf.placeholder("bool", shape=[])



#Gabor
pi = tf.asin(1.0) * 2.0
Theta = tf.Variable(tf.linspace(0.0, pi * 0.75, 4))
#print Theta.eval()
Lambda = tf.Variable(tf.linspace(2.0, 6.0, 4))
#print Lambda.eval()
Fai = tf.constant([0.0])
Sigma = 0.56 * Lambda
Gamma = tf.constant([1.0])
Gabor = Gabor_filter(Theta, Lambda, Fai, Sigma, Gamma, 17, 3, 16)





Gabor_conv = tf.nn.conv2d(x, Gabor, strides=[1, 1, 1, 1], padding='SAME')
x_ = tf.concat([x, Gabor_conv], 3)

W_conv1 = weight_variable([3, 3, 16, 36])
b_conv1 = bias_variable([36])

conv1 = convolution_2d(x, W_conv1, b_conv1)
act1 = tf.nn.relu(batch_norm()(conv1, train = BN_train))
pool1 = max_pooling(act1, 2, 2)
drop1 = tf.nn.dropout(pool1, keep_prob)

W_conv2 = weight_variable([3, 3, 36, 128])
b_conv2 = bias_variable([128])

conv2 = convolution_2d(drop1, W_conv2, b_conv2)
act2 = tf.nn.relu(batch_norm()(conv2, train = BN_train))
pool2 = max_pooling(act2, 2, 2)
drop2 = tf.nn.dropout(pool2, keep_prob)

W_conv3 = weight_variable([3, 3, 128, 256])
b_conv3 = bias_variable([256])

conv3 = convolution_2d(drop2, W_conv3, b_conv3)
act3 = tf.nn.relu(batch_norm()(conv3, train = BN_train))
pool3 = max_pooling(act3, 2, 2)
drop3 = tf.nn.dropout(pool3, keep_prob)

W_fc1 = weight_variable([4 * 4 * 256, 256])
b_fc1 = bias_variable([256])

flat = tf.reshape(drop3, [-1, 4 * 4 * 256])
fc1 = tf.nn.relu(batch_norm()((tf.matmul(flat, W_fc1) + b_fc1), train=BN_train))
drop = tf.nn.dropout(fc1, keep_prob)

W_fc2 = weight_variable([256, 10])
b_fc2 = bias_variable([10])

y_predict = tf.nn.softmax(tf.matmul(drop, W_fc2) + b_fc2)

cross_entropy = -tf.reduce_sum(y * tf.log(tf.clip_by_value(y_predict, 1e-10, 1.0)))
train_step = tf.train.AdamOptimizer(rate).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_predict, 1), tf.argmax(y, 1))

correct_num = tf.reduce_sum(tf.cast(correct_prediction, "float"))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

sess.run(tf.global_variables_initializer())







batch_image = np.zeros([batch_size,32,32,3])
batch_label = np.zeros([batch_size,10])
f = open('CNN_add_gabor.txt', 'a')
for i in range(100):
    num = 0
    total = 0
    learn_rate = 1e-3
    for j in range(50000 / batch_size):
        batch_image = train_data[j*batch_size : (j+1)*batch_size, :, :, :]
        batch_label = train_label[j*batch_size : (j+1)*batch_size, :]

        train_step.run(feed_dict={x: batch_image, y: batch_label, keep_prob: 0.5, BN_train: True, rate: learn_rate})
        num = num + correct_num.eval(feed_dict={x: batch_image, y: batch_label, keep_prob: 1.0, BN_train: False, rate: learn_rate})
        total = total + batch_size

        if j%50 == 49:
            print "step %d, %d / 50000, training accuracy %f"%(i, total, num/total)
            #print W_conv1.eval()
            #print Theta.eval()
            #print Lambda.eval()
            """
            plt.figure("show")
            img = np.zeros([32, 32, 3])
            show = np.zeros([32, 32, 3])
            img = batch_image[7, :, :, :]
            plt.imshow(img)
            plt.show()
            for t in range(36):
                tmp = Gabor.eval(feed_dict={x: batch_image, y: batch_label, keep_prob: 1.0})
                img0 = np.zeros([17, 17])
                img0 = tmp[:, :, 0, t]
                plt.imshow(img0, cmap = cm.gray)
                plt.show()
            """


    f.write("step %d, training accuracy %f"%(i, num/total))
    f.write('\n')

    num = 0
    total = 0

    for j in range(10000 / batch_size):
        batch_image = test_data[j*batch_size : (j+1)*batch_size, :, :, :]
        batch_label = test_label[j*batch_size : (j+1)*batch_size, :]

        num = num + correct_num.eval(feed_dict={x: batch_image, y: batch_label, keep_prob: 1.0, BN_train: False, rate: learn_rate})
        total = total + batch_size

    print "step %d, %d / 10000, test accuracy %f"%(i, total, num/total)
    if num/total > 0.7:
        learn_rate = 1e-4
    f.write("step %d, test accuracy %f" % (i, num / total))
    f.write('\n')
    print learn_rate




