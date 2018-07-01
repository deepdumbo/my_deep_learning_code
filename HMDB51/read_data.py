# -*- coding:utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import random

def read_data(rate = 0.8):
    depth = 40
    height = 32
    width = 40


    train_data = np.zeros([5068, depth, height, width, 3])
    train_label = np.zeros([5068, 51])
    test_data = np.zeros([1306, depth, height, width, 3])
    test_label = np.zeros([1306, 51])

    PATH = 'data/image'
    path_action = os.listdir(PATH + '/train')

    batch_train_num = 0
    batch_test_num = 0
    action_num = 0
    for action in path_action:
        path_video = os.listdir(PATH+'/train/'+action)
        num = len(path_video)
        print action, num
        for video in path_video:
            for i in range(depth):
                img = cv2.imread(PATH + '/train/' + action + '/' + video + '/' + str(i) + '.jpg')
                img = cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)
                #print np.shape(img)
                #cv2.imshow('show', img)
                #cv2.waitKey(300)
                train_data[batch_train_num, i, :, :, :] = img[:, :, :] # gray image

            train_label[batch_train_num, action_num] = 1  # one hot
            batch_train_num = batch_train_num + 1

        path_video = os.listdir(PATH + '/test/' + action)
        num = len(path_video)
        print action, num
        for video in path_video:
            for i in range(depth):
                img = cv2.imread(PATH + '/test/' + action + '/' + video + '/' + str(i) + '.jpg')
                img = cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)
                # print np.shape(img)
                # cv2.imshow('show', img)
                # cv2.waitKey(300)
                test_data[batch_test_num, i, :, :, :] = img[:, :, :]  # gray image

            test_label[batch_test_num, action_num] = 1  # one hot
            batch_test_num = batch_test_num + 1

        action_num = action_num + 1
    print 'train num:', batch_train_num
    print 'test num:', batch_test_num



    return train_data, train_label, test_data, test_label

if __name__ == '__main__':
    train_data, train_label, test_data, test_label = read_data()
    PATH = 'data/numpy/'

    np.save(PATH + "train_data.npy", train_data)
    np.save(PATH + "train_label.npy", train_label)
    np.save(PATH + "test_data.npy", test_data)
    np.save(PATH + "test_label.npy", test_label)





    #img = np.zeros([30, 40])
    #for i in range(300, 330):

        #print i, train_label
        #for j in range(40):
        #img[:, :] = train_data[i, 30, :, :, 0]
            #plt.figure("show")
            #plt.imshow(img, cmap ='gray')
            #plt.show()
        #cv2.imwrite(str(i)+'__'+str(train_label[i])+'test.jpg', img)
