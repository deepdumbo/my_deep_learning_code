import tensorflow as tf
import numpy as np
import cv2
import os
import time
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

class MyDataset(object):
    def __init__(self, PATH, batch_size):
        self.batch_size = batch_size
        self.train_get_next_cnt = 0
        self.test_get_next_cnt = 0

        self.train_data = []
        self.train_label = []
        self.test_data = []
        self.test_label = []

        train_data_list = os.listdir(PATH + '/numpy/train')
        self.train_num = len(train_data_list)
        for i in range(self.train_num):
            self.train_data.append(PATH + '/numpy/train/' + train_data_list[i] + '/data.npy')
            self.train_label.append(np.load(PATH + '/numpy/train/' + train_data_list[i] + '/label.npy'))

        test_data_list = os.listdir(PATH + '/numpy/test')
        self.test_num = len(test_data_list)
        for i in range(self.test_num):
            self.test_data.append(PATH + '/numpy/test/' + test_data_list[i] + '/data.npy')
            self.test_label.append(np.load(PATH + '/numpy/test/' + test_data_list[i] + '/label.npy'))

        self.train_data = np.array(self.train_data)
        self.train_label = np.array(self.train_label)
        self.test_data = np.array(self.test_data)
        self.test_label = np.array(self.test_label)
        self.shuffle()

        print(self.train_num, self.test_num)

    def shuffle(self):
        seed = np.random.randint(10000)
        np.random.seed(seed)
        np.random.shuffle(self.train_data)
        np.random.seed(seed)
        np.random.shuffle(self.train_label)

        seed = np.random.randint(10000)
        np.random.seed(seed)
        np.random.shuffle(self.test_data)
        np.random.seed(seed)
        np.random.shuffle(self.test_label)

    def train_get_next(self):
        if self.train_get_next_cnt + self.batch_size > self.train_num:
            self.train_get_next_cnt = 0
            self.shuffle()
        data = self.load_prestored_data(self.train_data[self.train_get_next_cnt: self.train_get_next_cnt + self.batch_size])
        label = self.train_label[self.train_get_next_cnt: self.train_get_next_cnt + self.batch_size]
        self.train_get_next_cnt += self.batch_size
        return data, label

    def test_get_next(self):
        if self.test_get_next_cnt + self.batch_size > self.test_num:
            self.test_get_next_cnt = 0
            self.shuffle()
        data = self.load_prestored_data(self.test_data[self.test_get_next_cnt: self.test_get_next_cnt + self.batch_size])
        label = self.test_label[self.test_get_next_cnt: self.test_get_next_cnt + self.batch_size]
        self.test_get_next_cnt += self.batch_size
        return data, label

    def load_prestored_data(self, PATH):
        batch_size = len(PATH)
        data = []

        for i in range(batch_size):
            data.append(np.load(PATH[i]))

        return np.array(data)