# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 16:52:12 2020

Copy right

@author: Saifut khan
version 1: read orignal images, convert to gray, resize and write back to folder that will contain
           the final images to be used in training/testing 
version2:  Produce labels from image names
           Convert images to 4D numeric tensor and labels to 2D numeric matrix
version 3: Build a 5 layer CNN for digit recognition. 
           Load data, run model and get train and test accuracy
    
"""
#import cv2
import os
import numpy as np
import imageio
import math
from keras.models import Sequential
from keras.layers.core import Dense, Flatten
from keras.layers.convolutional import Conv2D
from keras.optimizers import Adam
from keras.layers.pooling import MaxPooling2D
from keras.utils import to_categorical
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

#goodpath="D:\\capstone\\ai\\data\\good"
#badpath="D:\\capstone\\ai\\data\\bad"
finalpath="D:\\capstone\\ai\\data\\final"
   
# DO NOT DELETE THIS LINE haarhand=cv2.CascadeClassifier('Hand_haar_cascade.xml')
def finalImgs(rpath,wpath):
    for f in os.listdir(rpath):
        rf=os.path.join(rpath,f)
        img=cv2.imread(rf)
        gry=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        gray=cv2.resize(gry,(64,64),interpolation=cv2.INTER_AREA)
        wf=os.path.join(wpath,f)
        cv2.imwrite(wf,gray)

def loadData(X,Y):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.7, stratify=Y)
    return X_train, X_test, Y_train, Y_test
    
   
def createDataSets(imgpath):
    imgs=[]
    labels=[]
    sortedfiles=sorted(os.listdir(imgpath))

    for f in sortedfiles:
        sections=[]
        rf=os.path.join(imgpath,f)
        sections=f.split('.')
        if sections[1] == 'good':
            labels.append(sections[0])
        else:
            labels.append('bad')
            
        img=imageio.imread(rf)
        imgs.append(img)
    
    return imgs, labels
    
# generate final good xformed images
#finalImgs(goodpath,finalpath)
imgs, labels=createDataSets(finalpath)
X_train, X_test, Y_train, Y_test = loadData(np.array(imgs), np.array(labels))
X_train = X_train/255
X_test = X_test/255
print(X_train.shape)
print(X_test.shape)

# transform traing and testing labels to one hot encoding
le = preprocessing.LabelEncoder()             # need to use this
le.fit(Y_train)
y=le.transform(Y_train)
Y_train=to_categorical(y)

le = preprocessing.LabelEncoder()             # need to use this
le.fit(Y_test)
y=le.transform(Y_test)
Y_test=to_categorical(y)

print(Y_train.shape)
print(Y_test.shape)


# Build CNN model
model = Sequential()
model.add(Conv2D(128,kernel_size=(3,3),activation='relu',input_shape=(64,64,3)))
model.add(Conv2D(128,kernel_size=(3,3),activation='relu'))
model.add(Conv2D(128,kernel_size=(3,3),activation='relu'))
model.add(Conv2D(128,kernel_size=(3,3),padding='same',activation='relu'))
model.add(Flatten())
model.add(Dense(Y_train.shape[1],activation='softmax'))
model.compile(optimizer='Adam',loss='categorical_crossentropy', metrics=['accuracy'])

# Run model and evalaute accuracy
history=model.fit(X_train,Y_train,batch_size=256,epochs=3,validation_data=(X_test, Y_test))
scores = model.evaluate(X_test, Y_test)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# print accuracy
print('Loss: %.3f' % scores[0])
print('Accuracy: %.3f' % scores[1])




