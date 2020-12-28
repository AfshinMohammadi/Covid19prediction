import os
import cv2
import  tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Dense, MaxPooling2D, Conv2D, Flatten, Dropout
from keras.models import Sequential
import tensorflow as tf
from keras.utils import to_categorical
import tensorboard

posfiles = os.listdir('G:/cshub.mit/python/Projects/Covid_data/train/pos')
negfiles = os.listdir('G:/cshub.mit/python/Projects/Covid_data/train/neg')


classes = ['pos', 'neg']
pos = []
neg = []
j = 32
p = 32
x = []
y = []


for clas in classes:
    if clas == 'pos':
        for i in posfiles:
            t = cv2.imread('G:/cshub.mit/python/Projects/Covid_data/train/pos' + '/' + i)
            q = cv2.resize(t, (j, p))
            x.append(q)
            y.append(0)
    if clas == 'neg':
        for q in negfiles:
            t = cv2.imread('G:/cshub.mit/python/Projects/Covid_data/train/neg' + '/' + q)
            q = cv2.resize(t, (j, p))
            x.append(q)
            y.append(1)
    


x = np.asarray(x)
x = x/255
x = x.astype('float32')
xtrain = tf.convert_to_tensor(x)
n = np.asarray(y)
y = n.astype('float32')
y = tf.convert_to_tensor(y)
ytrain = to_categorical(y, num_classes=2, dtype='float32')


model = Sequential()
model.add(Conv2D(128, kernel_size=(5,5), activation='relu', input_shape=(32, 32, 3)))
model.add(MaxPooling2D(pool_size=(4, 4)))
model.add(Dropout(0.2))
model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Conv2D(32, kernel_size=(2,2), activation='relu'))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(128, activation = 'relu'))
model.add(Dropout(0.25))
model.add(Dense(64, activation = 'relu'))
model.add(Dropout(0.25))
model.add(Dense(2, activation='sigmoid'))


model.compile(optimizer='adam', loss='BinaryCrossentropy', metrics=['accuracy'])
model.fit(xtrain, ytrain, batch_size=1, epochs = 20, verbose = 2, validation_split=0.1)

posfiles = os.listdir('G:/cshub.mit/python/Projects/Covid_data/test/pos')
negfiles = os.listdir('G:/cshub.mit/python/Projects/Covid_data/test/neg')
pos = []
neg = []
x = []
y = []


for clas in classes:
    if clas == 'pos':
        for i in posfiles:
            t = cv2.imread('G:/cshub.mit/python/Projects/Covid_data/test/pos' + '/' + i)
            q = cv2.resize(t, (j, p))
            x.append(q)
            y.append(0)
    if clas == 'neg':
        for q in negfiles:
            t = cv2.imread('G:/cshub.mit/python/Projects/Covid_data/test/neg' + '/' + q)
            q = cv2.resize(t, (j, p))
            x.append(q)
            y.append(1)
    


x = np.asarray(x)
x = x/255
x = x.astype('float32')
xtest = tf.convert_to_tensor(x)
y = np.asarray(y)
y = y.astype('float32')
y = tf.convert_to_tensor(y)
ytest = to_categorical(y, num_classes=2, dtype='float32')


ypre = model.predict(xtest)
m = model.predict_classes(xtest)
p = model.evaluate(xtest, ytest)