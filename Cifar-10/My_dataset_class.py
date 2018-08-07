import tensorflow as tf
import pickle as p
import numpy as np
import cv2
import os
import time
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

class MyDataset(object):
    def __init__(self, PATH, batch_size):
        self.PATH = PATH
        self.batch_size = batch_size
        self.data = np.zeros([self.batch_size, 32, 32, 3])
        self.label = np.zeros([self.batch_size, 10])
        self.train_get_next_cnt = 0
        self.test_get_next_cnt = 0

        self.train_data, self.train_label, self.test_data, self.test_label = self.load_CIFAR()

        self.train_num = len(self.train_data)
        self.test_num = len(self.test_data)
        self.shuffle()
        print(self.train_num, self.test_num)

    def shuffle(self):
        seed = np.random.randint(10000)
        np.random.seed(seed)
        np.random.shuffle(self.train_data)
        np.random.seed(seed)
        np.random.shuffle(self.train_label)

    def train_get_next(self):
        if self.train_get_next_cnt + self.batch_size > self.train_num:
            self.train_get_next_cnt = 0
            self.shuffle()

        self.data[:, :, :, :] = self.train_data[self.train_get_next_cnt: self.train_get_next_cnt+self.batch_size, :, :, :]
        self.label[:, :] = self.train_label[self.train_get_next_cnt: self.train_get_next_cnt+self.batch_size, :]
        self.train_get_next_cnt += self.batch_size
        return self.data, self.label

    def test_get_next(self):
        if self.test_get_next_cnt + self.batch_size > self.test_num:
            self.test_get_next_cnt = 0
        self.data[:, :, :, :] = self.test_data[self.test_get_next_cnt: self.test_get_next_cnt + self.batch_size, :, :, :]
        self.label[:, :] = self.test_label[self.test_get_next_cnt: self.test_get_next_cnt + self.batch_size, :]
        self.test_get_next_cnt += self.batch_size
        return self.data, self.label

    def load_CIFAR_batch(self, filename):
        """ load single batch of cifar """
        with open(filename, 'rb')as f:
            datadict = p.load(f, encoding='iso-8859-1')
            X = datadict['data']
            Y = datadict['labels']
            X = X.reshape(10000, 3, 32, 32)
            Y = np.array(Y)
            return X, Y

    def load_CIFAR(self):
        train_data = np.zeros([50000, 32, 32, 3])
        train_label = np.zeros([50000, 10])
        for sample in range(5):
            X, Y = self.load_CIFAR_batch(self.PATH + "/data_batch_" + str(sample + 1))

            for i in range(3):
                train_data[10000 * sample:10000 * (sample + 1), :, :, i] = X[:, i, :, :]
            for i in range(10000):
                train_label[i + 10000 * sample][Y[i]] = 1

        test_data = np.zeros([10000, 32, 32, 3])
        test_label = np.zeros([10000, 10])
        X, Y = self.load_CIFAR_batch(self.PATH + "/test_batch")
        for i in range(3):
            test_data[0:10000, :, :, i] = X[:, i, :, :]
        for i in range(10000):
            test_label[i][Y[i]] = 1

        return train_data, train_label, test_data, test_label