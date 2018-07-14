# -*- coding:utf-8 -*-

import numpy as np
import cv2
import os
import shutil

PATH = 'data/video'
path_action = os.listdir(PATH)
for action in path_action:
    print(action)
    path_video = os.listdir(PATH + '/' + action)
    train_cnt = 0
    test_cnt = 0

    test_random = np.random.randint(5) # 20% for the test data

    for video in path_video:
        test_random = np.random.randint(5)  # 20% for the test data
        if test_random == 0:
            test_cnt = test_cnt + 1
            # print video
            cap = cv2.VideoCapture(PATH + '/' + action + '/' + video)
            for i in range(40):  # 5 frame for 3dCNN and 8 step for lstm
                ret, frame = cap.read()
                if ret:
                    path_save = 'data/image/test/' + action + '/' + str(test_cnt) + '/'
                    if not os.path.exists(path_save):
                        os.makedirs(path_save)
                    cv2.imwrite(path_save + str(i) + '.jpg', frame)
                    # cv2.imshow("capture", frame)
                else:
                    print('error:' + video + 'only have ' + str(i) + 'frames!')
                    shutil.rmtree(path_save)
                    test_cnt = test_cnt - 1
                    break
        else:
            train_cnt = train_cnt + 1
            # print video
            cap = cv2.VideoCapture(PATH + '/' + action + '/' + video)
            for i in range(40):  # 5 frame for 3dCNN and 12 step for lstm
                ret, frame = cap.read()
                if ret:
                    path_save = 'data/image/train/' + action + '/' + str(train_cnt) + '/'
                    if not os.path.exists(path_save):
                        os.makedirs(path_save)
                    cv2.imwrite(path_save + str(i) + '.jpg', frame)
                    # cv2.imshow("capture", frame)
                else:
                    print('error:' + video + 'only have ' + str(i) + 'frames!')
                    shutil.rmtree(path_save)
                    train_cnt = train_cnt - 1
                    break

    #print action