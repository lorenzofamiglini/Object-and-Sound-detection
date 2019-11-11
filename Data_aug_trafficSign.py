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
"""
Itera per tutta la cartella numerata con ogni elemento della variabile target e ne estrae il campione di immagini per ciascuna etichetta.
Successivamente salvami il valore della matrice della foto in una lista mentre in un altra la classe della variabile target associata
"""
for i in range(0,classes):
    path = "/Users/lorenzofamiglini/Desktop/MsC_2_anno/PROGETTO_AML_DSIM/gtsrb-german-traffic-sign/Train/{0}/".format(i)
    Class=os.listdir(path)
    for a in Class:
        image=cv2.imread(path+a)
        image_from_array = Image.fromarray(image, 'RGB')
        #size_image = image_from_array.resize((height, width))
        data.append(np.array(image_from_array))
        labels.append(str(i))

X_train =np.array(data)
y_train=np.array(labels)
#Converto y_train da stringa a valore intero
y_train = y_train.astype(np.int)
len(data)
len(y_train)
np.unique(y_train)
np.shape(data)

"""
ANALISI DATI
"""
count_label = pd.DataFrame(y_train,  columns = ["Target"])
count_label

df = pd.read_csv("/Users/lorenzofamiglini/Desktop/MsC_2_anno/PROGETTO_AML_DSIM/Object-and-Sound-detection/count_target.csv")
df.shape
df_merge = pd.merge(count_label, df, left_on = "Target", right_on = "target", how = "right")
df_merge.drop(['Target', 'Unnamed: 0'], axis=1, inplace = True)
count_df = df_merge.groupby(['title']).count()

import seaborn as sns
sns.set(style="darkgrid")
count_df = count_df.reset_index()

from matplotlib import pyplot as plt
from matplotlib import style

style.use('ggplot')

plt.figure(figsize=(10,10))
plt.barh("title","type", data = count_df,align='center', alpha=0.5)
plt.xlabel('Count')
plt.title("Observations count per class")
plt.show()

df_merge["type"].value_counts()

pd.DataFrame(df_merge["title"].value_counts()).sort_values


"""
Analisi della dimensione dei pixel
"""
lista_hei = [height.shape[0] for height in X_train]
lista_wid = [width.shape[1] for width in X_train]
df_hw = pd.DataFrame()
df_hw["height"] = lista_hei
df_hw["width"] = lista_wid
df_hw["height"].mean()
df_hw["width"].mean()
g = sns.JointGrid(x="height", y="width", data=df_hw)
g.plot_joint(sns.regplot, order=2)
g.plot_marginals(sns.distplot)

g = sns.jointplot(x="height", y="width", data=df_hw, kind='kde')

"""
Data augmentation
"""
from keras.preprocessing import image
for i in range(0,10):
    x = X_train[i].reshape((1,)+ X_train[i].shape)
    x.shape


    datagen = keras.preprocessing.image.ImageDataGenerator(featurewise_center=True,
        featurewise_std_normalization=True,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=False)
    i = 0
    for batch in datagen.flow(x, batch_size=1):
        plt.figure(i)
        app = (batch - batch.min()) / (batch.max()-batch.min())
        #imgplot= plt.imshow(app[0])
        plt.imshow(cv2.cvtColor(app[0], cv2.COLOR_BGR2RGB))
        i += 1
        if i % 10 == 0:
            break
        plt.show()


"""
Normalizzazione
"""

X_train = (X_train-X_train.min())/(X_train.max()-X_train.min())
X_train = X_train.reshape(X_train.shape[0],30*30,3)
X_train.max()
X_train.min()



"""
Shuffle dataset
"""
from sklearn.utils import shuffle
y_train = y_train.reshape(y_train.shape[0])
X_train, y_train = shuffle(X_train,y_train, random_state=1)
