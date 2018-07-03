import keras
from keras.models import Sequential
from keras.utils import np_utils
from keras.layers import Dense, Dropout, Activation, MaxPooling2D, Flatten, Conv2D
from keras.datasets import mnist
from keras.optimizers import RMSprop
import pandas as pd
import numpy as np


(in_train, out_train), (in_test, out_test) = mnist.load_data()
in_train_MLP = in_train.reshape(60000, 784)
in_test_MLP = in_test.reshape(10000, 784)
in_train_MLP = in_train.astype('float32')
in_test_MLP = in_test.astype('float32')
in_train_MLP_normalize = in_train / 255
in_test_MLP_normalize = in_test / 255

out_train_onehot = np_utils.to_categorical(out_train)
out_test_onehot = np_utils.to_categorical(out_test)






in_train_CNN = in_train.reshape(in_train.shape[0], 28, 28, 1).astype('float32')  
in_test_CNN = in_test.reshape(in_test.shape[0], 28, 28, 1).astype('float32')
in_train_CNN_normalize = in_train / 255
in_test_CNN_normalize = in_test / 255

print('model2')
model2 = Sequential()
model2.add(Conv2D(8, (3, 3), input_shape = (28,28,1), padding='same', activation='relu'))
model2.add(Dropout(0.25))
model2.add(MaxPooling2D((2, 2)))

model2.add(Conv2D(16, (3, 3), padding='same', activation='relu'))
model2.add(Dropout(0.25))
model2.add(MaxPooling2D((2, 2)))

model2.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
model2.add(Dropout(0.25))
model2.add(MaxPooling2D((2, 2)))

model2.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model2.add(Dropout(0.25))
model2.add(MaxPooling2D((2, 2)))

model2.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
model2.add(Dropout(0.25))
model2.add(MaxPooling2D((2, 2)))

model2.add(Flatten())
model2.add(Dense(256, activation='relu'))
model2.add(Dropout(0.25))
model2.add(Dense(128, activation='relu'))
model2.add(Dropout(0.25))
model2.add(Dense(64, activation='relu'))
model2.add(Dropout(0.25))
model2.add(Dense(32, activation='relu'))
model2.add(Dropout(0.25))
model2.add(Dense(10, activation='softmax'))
model2.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print(model2.evaluate(in_train_CNN_normalize, out_test_onehot))
model2.fit(in_test_CNN_normalize, out_train_onehot, epochs = 15, batch_size = 196)

out_pred = model2.predict_classes(in_test)
print(pd.crosstab(out_test, out_pred))
