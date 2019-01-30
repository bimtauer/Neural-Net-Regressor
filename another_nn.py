# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 14:35:12 2019

@author: bimta
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf


import os, os.path, gzip, tempfile, urllib.request

def load_mnist(kind='train', dataset='zalando'): # 'train' or 't10k'
    """based on https://github.com/zalandoresearch/fashion-mnist/blob/master/utils/mnist_reader.py"""

    if dataset=='zalando':
        url_base = 'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/'
    else:
        url_base = 'http://yann.lecun.com/exdb/mnist/'

    url_labels = url_base+'%s-labels-idx1-ubyte.gz'%kind
    url_images = url_base+'%s-images-idx3-ubyte.gz'%kind

    file_labels = os.path.join(tempfile.gettempdir(), '%s-labels-idx1-ubyte.gz'%kind)
    file_images = os.path.join(tempfile.gettempdir(), '%s-images-idx3-ubyte.gz'%kind)

    if not os.path.exists(file_labels):
        urllib.request.urlretrieve(url_labels, file_labels)

    if not os.path.exists(file_images):
        urllib.request.urlretrieve(url_images, file_images)

    with gzip.open(file_labels, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)

    with gzip.open(file_images, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)

    assert len(images.shape)==2
    assert len(labels.shape)==1
    assert images.shape[0] == labels.shape[0]
    assert images.shape[1] == 28*28
    return images, labels

X_train, Y_train = load_mnist('train')
X_test, Y_test = load_mnist('t10k')

# Scale to 0-1
X_train /= 255
X_test /= 255

plt.imshow(X_train[34].reshape(28,28), cmap= 'gray')


# Faster
def one_hot_encode(y):
    enc = np.zeros((y.shape[0], len(set(y))))
    enc[np.arange(len(y)), y] = 1
    return enc

# Neater
def one_hot_encode2(y):
    return np.eye(np.max(y) + 1)[y,:]

def one_hot_decode2(y):
    lab = np.arange(y.shape[1]).astype('uint8') 
    return y @ lab

def one_hot_decode(y):
    return np.argmax(y, axis=1)

Y_enc = one_hot_encode(Y_test)

Y_dec = one_hot_decode(Y_enc)

(Y_dec == Y_test).all()

# Building our own knn


X_train_subset = X_train[:6000,:]
Y_train_subset = Y_train[:6000]
X_test_subset = X_test[:100,]
Y_test_subset = Y_test[:100]


def knn(X_train, Y_train, X_test, k):
    preds = []
    # For every test sample
    for row in X_test:
        # get the indices of the k closest elements in X_train 
        i = np.argsort(np.linalg.norm((X_train - row), axis=1), )[:k]
        # find the corresponding lables
        labels = Y_train[i]
        # majority vote for those
        vals, counts = np.unique(labels, return_counts=True)
        pred = np.random.choice(vals[counts == counts.max()], 1)
        preds.append(pred)
    return np.array(preds).ravel()

y_pred = knn(X_train_subset, Y_train_subset, X_test_subset, 5)        



# Accuracy
def accuracy(Y_pred, Y_test):
    return np.mean(Y_pred == Y_test)


accuracy(Y_test_subset, y_pred)

from sklearn.metrics import classification_report
print(classification_report(y_pred, Y_test_subset))


###############################################################################
#  Multinomial logistic regression aka 1-layer net









