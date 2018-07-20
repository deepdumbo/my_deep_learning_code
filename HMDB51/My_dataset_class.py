import tensorflow as tf
import numpy as np
import cv2
import os
import time
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

class MyDataset(object):
    def __init__(self, PATH, batch_size, proportion):
        self.batch_size = batch_size
        self.train_get_next_cnt = 0
        self.test_get_next_cnt = 0

        self.train = []
        self.test = []

        data_list = os.listdir(PATH)
        for data_name in data_list:
            if np.random.rand() < proportion:
                self.test.append(PATH + '/' + data_name)
            else:
                self.train.append(PATH + '/' + data_name)

        self.train_num = len(self.train)
        self.test_num = len(self.test)

        self.train = np.array(self.train)
        self.test = np.array(self.test)
        self.shuffle()

        print(self.train_num, self.test_num)

    def shuffle(self):
        seed = np.random.randint(10000)
        np.random.seed(seed)
        np.random.shuffle(self.train)
        np.random.seed(seed)
        np.random.shuffle(self.train)

    def train_get_next(self):
        if self.train_get_next_cnt + self.batch_size > self.train_num:
            self.train_get_next_cnt = 0
            self.shuffle()
        data, label = self.load_prestored_data(self.train[self.train_get_next_cnt: self.train_get_next_cnt + self.batch_size])
        self.train_get_next_cnt += self.batch_size
        return data, label

    def test_get_next(self):
        if self.test_get_next_cnt + self.batch_size > self.test_num:
            self.test_get_next_cnt = 0
            self.shuffle()
        data, label = self.load_prestored_data(self.test[self.test_get_next_cnt: self.test_get_next_cnt + self.batch_size])
        self.test_get_next_cnt += self.batch_size
        return data, label

    def load_prestored_data(self, PATH):
        batch_size = len(PATH)
        data = []
        label = []

        for i in range(batch_size):
            tmp = np.load(PATH[i])
            data.append(tmp[0])
            label.append(tmp[1])

        return np.array(data), np.array(label)