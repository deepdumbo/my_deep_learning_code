import cv2
import os
import numpy as np
import pickle as p

def load_CIFAR_batch(filename):
    """ load single batch of cifar """
    with open(filename, 'rb')as f:
        datadict = p.load(f,encoding='iso-8859-1')
        X = datadict['data']
        Y = datadict['labels']
        X = X.reshape(10000, 3, 32, 32)
        Y = np.array(Y)
        return X, Y


def load_CIFAR(Foldername):
    train_data = np.zeros([50000, 32, 32, 3])
    train_label = np.zeros([50000, 10])

    for sample in range(5):
        X, Y = load_CIFAR_batch(Foldername + "/data_batch_" + str(sample + 1))

        for i in range(3):
            train_data[10000 * sample:10000 * (sample + 1), :, :, i] = X[:, i, :, :]
        for i in range(10000):
            train_label[i + 10000 * sample][Y[i]] = 1

    test_data = np.zeros([10000, 32, 32, 3])
    test_label = np.zeros([10000, 10])
    X, Y = load_CIFAR_batch(Foldername + "/test_batch")
    for i in range(3):
        test_data[0:10000, :, :, i] = X[:, i, :, :]
    for i in range(10000):
        test_label[i][Y[i]] = 1

    return train_data, train_label, test_data, test_label

def cifar_save_as_jpg(Foldername):
    try:
        os.mkdir('train')
        os.mkdir('test')
        for i in range(10):
            os.mkdir('train/'+str(i)+'/')
            os.mkdir('test/'+str(i)+'/')
    except:
        pass

    img = np.zeros([32, 32, 3])
    for sample in range(5):
        X, Y = load_CIFAR_batch(Foldername + "/data_batch_" + str(sample + 1))
        for i in range(10000):
            for c in range(3):
                img[:, :, c] = X[i, c, :, :]
            cv2.imwrite('train/'+str(Y[i])+'/'+str(sample*10000+i).zfill(5)+'.jpg', img)
    X, Y = load_CIFAR_batch(Foldername + "/test_batch")
    for i in range(10000):
        for c in range(3):
            img[:, :, c] = X[i, c, :, :]
        cv2.imwrite('test/'+str(Y[i]) + '/' + str(i) + '.jpg', img)

if __name__ == '__main__':
    cifar_save_as_jpg('ori_data')
