import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import pandas as pd
import os
import sys
import time
import tensorflow as tf
from tensorflow import keras
import cv2
print(tf.__version__)
print(sys.version_info)
for module in mpl, np, pd, sklearn, tf, keras:
    print(module.__name__, module.__version__)
fashion_mnist = keras.datasets.fashion_mnist  #加载TensorFlow中自带的数据集

#拆分数据集，加载数据集后返回训练集以及测试集
(x_train_all, y_train_all), (x_test, y_test) = fashion_mnist.load_data()

#将训练集进行一次拆分为验证集和训练集
x_valid, x_train = x_train_all[:5000], x_train_all[5000:]
y_valid, y_train = y_train_all[:5000], y_train_all[5000:]

print(x_valid.shape, y_valid.shape)
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)
print(x_valid[0].shape)
cv2.imshow('name' , x_valid[0])
cv2.waitKey(0)
cv2.imshow('name' , x_valid[1])
cv2.waitKey(0)
