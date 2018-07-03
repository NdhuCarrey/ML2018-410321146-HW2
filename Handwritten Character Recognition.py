import keras
from keras.models import Sequential
from keras.utils import np_utils
from keras.layers import Dense, Dropout, Activation, LeakyReLU, Conv2D
from keras.datasets import mnist
from keras.optimizers import RMSprop
import pandas as pd
import numpy as np


(in_train, out_train), (in_test, out_test) = mnist.load_data()
in_train = in_train.reshape(60000, 784)
in_test = in_test.reshape(10000, 784)
in_train = in_train.astype('float32')
in_test = in_test.astype('float32')
in_train_normalize /= 255
in_test_normalize /= 255


