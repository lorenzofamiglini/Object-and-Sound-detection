import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import os


"""
Importiamo tutte le immagini dalle varie cartelle
"""

data=[]
labels=[]
for i in range(0,classes):
    print(i)

height = 30
width = 30
channels = 3
classes = 43
n_inputs = height * width*channels

for i in range(0,classes):
    print(i)
    path = "/Users/lorenzofamiglini/Downloads/gtsrb-german-traffic-sign/Train/{0}/".format(i)
    Class=os.listdir(path)
    for a in Class:
        image=cv2.imread(path+a)
        image_from_array = Image.fromarray(image, 'RGB')
        size_image = image_from_array.resize((height, width))
        data.append(np.array(size_image))
        labels.append(str(i))

X_train =np.array(data)
y_train=np.array(labels)
y_train
len(data)
len(y_train)
np.unique(y_train)
"""
Shuffle dataset
"""




plt.axis("off")
plt.title(y_train[3])
plt.imshow(cv2.cvtColor(X_train[3], cv2.COLOR_BGR2RGB))
plt.show()
