import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import os
import keras

"""
Importiamo tutte le immagini dalle varie cartelle
"""

data=[]
labels=[]

height = 30
width = 30
channels = 3
classes = 43
n_inputs = height * width*channels

for i in range(len(X_train[0:3])):
    plt.axis("off")
    plt.title(y_train[i])
    plt.imshow(cv2.cvtColor(X_train[i], cv2.COLOR_BGR2RGB))
    plt.show()

for i in range(0,classes):
    path = "/Users/lorenzofamiglini/Desktop/MsC_2_anno/PROGETTO_AML_DSIM/gtsrb-german-traffic-sign/Train/{0}/".format(i)
    Class=os.listdir(path)
    for a in Class:
        image=cv2.imread(path+a)
        image_from_array = Image.fromarray(image, 'RGB')
        size_image = image_from_array.resize((height, width))
        data.append(np.array(size_image))
        labels.append(str(i))

X_train =np.array(data)
y_train=np.array(labels)

len(data)
len(y_train)
np.unique(y_train)
np.shape(data)
"""
Shuffle dataset
"""
from sklearn.utils import shuffle
y_train = y_train.reshape(y_train.shape[0])
X_train, y_train = shuffle(X_train,y_train, random_state=1)

for i in range(len(X_train[0:100])):
    plt.axis("off")
    plt.title(y_train[i])
    plt.imshow(cv2.cvtColor(X_train[i], cv2.COLOR_BGR2RGB))
    plt.show()

"""
Data augmentation
"""
from keras.preprocessing import image
x = X_train[0].reshape((1,)+ X_train[0].shape)
x.shape


datagen = keras.preprocessing.image.ImageDataGenerator(featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)



i = 0

for batch in datagen.flow(x, batch_size=1):
    plt.figure(i)
    imgplot= plt.imshow(image.array_to_img(batch[0]))
    i += 1
    if i % 100 == 0:
        break
plt.show()


plt.imshow(X_train[0], cmap="gray")
plt.axis('off')
plt.show()











"""
Normalizzazione
"""

X_train = (X_train-X_train.min())/(X_train.max()-X_train.min())
X_train = X_train.reshape(X_train.shape[0],30*30,3)

"""
Denoising autoencoder
"""

from keras.layers import Input, Dense
from keras.models import Model

input_img= Input(shape=(np.shape(X_train)[1]*np.shape(X_train)[1],3))

# encoded and decoded layer for the autoencoder
encoded = Dense(units=128, activation='relu')(input_img)
encoded = Dense(units=64, activation='relu')(encoded)
encoded = Dense(units=32, activation='relu')(encoded)
decoded = Dense(units=64, activation='relu')(encoded)
decoded = Dense(units=128, activation='relu')(decoded)
decoded = Dense(units=900, activation='sigmoid')(decoded)

# Building autoencoder
autoencoder=Model(input_img, decoded)
#extracting encoder
encoder = Model(input_img, encoded)
# compiling the autoencoder
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=['accuracy'])
# Fitting the noise trained data to the autoencoder
autoencoder.fit(X_train, X_train,
                epochs=100,
                batch_size=256,
                shuffle=True, verbose=1)
