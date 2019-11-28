import pandas as pd
import numpy as np
import keras
import matplotlib.pyplot as plt
import IPython.display as ipd
import scipy as sy
from scipy.io import wavfile as wav #importare ed esportare file audio
import pylab
import os
import sys
"""
Importo i audio
"""

giox_path = "/Users/lorenzofamiglini/Desktop/Test_Audio/Giorgio_set/"
lox_path = "/Users/lorenzofamiglini/Desktop/Test_Audio/Lorenzo_set/"
#Descrizione + chi ha parlato Lorenzo
lox_descr = np.load(lox_path+"lor_audio_descrizione.npy")
lox_y_descr_de = np.repeat("descrizione", len(lox_descr))
lox_y_descr_chi = np.repeat("lorenzo", len(lox_descr))
#Sanzione + chi ha parlato Lorenzo
lox_sanz = np.load(lox_path+"lor_audio_sanzione.npy")
lox_y_sanz_sa = np.repeat("sanzione", len(lox_sanz))
lox_y_sanz_chi = np.repeat("lorenzo", len(lox_sanz))
# Tipologia + chi ha parlato Lorenzo
lox_tipo = np.load(lox_path+"lor_audio_tipologia.npy")
lox_y_tipo_ti = np.repeat("tipologia", len(lox_tipo))
lox_y_tipo_chi = np.repeat("lorenzo", len(lox_tipo))

#Descrizione + chi ha parlato Giorgio
gio_descr = np.load(giox_path+"giox_audio_descrizione.npy")
gio_y_descr_de = np.repeat("descrizione", len(gio_descr))
gio_y_descr_chi = np.repeat("giorgio", len(gio_descr))
#Sanzione + chi ha parlato Giorgio
gio_sanz = np.load(giox_path+"giox_audio_sanzione.npy")
gio_y_sanz_sa = np.repeat("sanzione", len(gio_sanz))
gio_y_sanz_chi = np.repeat("giorgio", len(gio_sanz))
# Tipologia + chi ha parlato Giorgio
gio_tipo = np.load(giox_path+"giox_audio_tipologia.npy")
gio_y_tipo_ti = np.repeat("tipologia", len(gio_tipo))
gio_y_tipo_chi = np.repeat("giorgio", len(gio_tipo))

"""
Uniamo tutti gli audio con le giuste labels y
"""

X = np.concatenate([lox_descr, gio_descr, lox_sanz, gio_sanz, lox_tipo, gio_tipo], axis = 0)
y_cosa = np.concatenate([lox_y_descr_de, gio_y_descr_de, lox_y_sanz_sa, gio_y_sanz_sa, lox_y_tipo_ti, gio_y_tipo_ti], axis = 0)
y_chi = np.concatenate([lox_y_descr_chi, gio_y_descr_chi, lox_y_sanz_chi, gio_y_sanz_chi, lox_y_tipo_chi, gio_y_tipo_chi], axis = 0)


"""
Audio augmentation
"""
#Inserimento del rumore gaussiano
y_noise_aug = []
x_copy = X
for i in x_copy:
  y_noise = i
  noise_amp = 0.08*np.random.uniform()*np.amax(y_noise)
  y_noise = y_noise.astype('float64') + noise_amp * np.random.normal(size=y_noise.shape[0])
  y_noise_aug.append(y_noise)


from IPython.display import Audio
Audio(y_noise_aug[26], rate=44100)
Audio(X[26], rate=44100)

plt.figure(1)
plt.figure(figsize=(10,10))
#plt.title("Signal Wave...")
plt.plot(y_noise_aug[1], color = "red")
plt.plot(X[1],  color = "gray", alpha = 0.9)
#plt.title("Signal Wave...")
plt.show()


#Random shifting
y_shift_aug_dx = []
for i in x_copy:
    y_shift = i
    timeshift_fac = 0.3*(np.random.uniform()-0.2)  # up to 20% of length
    start = int(y_shift.shape[0] * timeshift_fac)
    if (start > 0):
        y_shift = np.pad(y_shift,(start,0),mode='constant')[0:y_shift.shape[0]]
        y_shift_aug_dx.append(y_shift)
    else:
        y_shift = np.pad(y_shift,(-start,0),mode='constant')[0:y_shift.shape[0]]
        y_shift_aug_dx.append(y_shift)


plt.figure(figsize=(10,10))
#plt.title("Signal Wave...")
plt.plot(X[1], color = "red")
plt.plot(y_shift_aug_dx[1], color = "gray", alpha = 0.9)
plt.show()

Audio(y_shift_aug_dx[1], rate=44100)
Audio(X[1], rate=44100)


new_x = np.concatenate([X, np.array(y_noise_aug), np.array(y_shift_aug_dx)], axis = 0)
new_y_chi = np.concatenate([y_chi, y_chi, y_chi], axis = 0)
new_y_cosa = np.concatenate([y_cosa, y_cosa, y_cosa], axis = 0)
df = pd.DataFrame(new_y_cosa)
import seaborn as sns
sns.countplot(x=0, data=df)

"""
FEATURE EXTRACTION
"""

import librosa
"""
def zcr(input):
    return (np.sum(np.abs(np.diff(np.sign(input),1)),keepdims=True))/2
def sdev(input):
    return np.std(input, keepdims=True) #keepdims mantiene il vettore sotto forma di array
def Energy(input):
    return np.sum((input*1)**2, keepdims=True)
def shortTermEnergy(input):
  return sum( [abs(x)**2 for x in input] ) / len(input)

def modified_spec(x):
  x = librosa.stft(x)
  xdb = librosa.amplitude_to_db(abs(x), )
  return xdb

sdev_train = np.array([sdev(i) for i in new_x])
energy_train = np.array([Energy(i) for i in new_x])
shorEn_train = np.array([shortTermEnergy(i) for i in new_x])

chroma_stft = np.array([librosa.feature.chroma_stft(y=i, sr=44100) for i in new_x])
rmse = np.array([librosa.feature.rms(y=i) for i in new_x])
spec_cent = np.array([librosa.feature.spectral_centroid(y=i, sr=44100) for i in new_x])
spec_bw = np.array([librosa.feature.spectral_bandwidth(y=i, sr=44100) for i in new_x])
rolloff = np.array([librosa.feature.spectral_rolloff(y=i, sr=44100) for i in new_x])
zcr = np.array([librosa.feature.zero_crossing_rate(y=i) for i in new_x])
mfcc = np.array([librosa.feature.mfcc(y=i, sr=44100) for i in new_x])
spec = np.array([modified_spec(i) for i in new_x])
"""


def feat_prova(X):
    sample_rate = 44100
    # short term fourier transform: represents a signal in the time-frequency domain by computing discrete Fourier transforms (DFT) over short overlapping windows.
    stft = np.abs(librosa.stft(X))

    # mfcc (mel-frequency cepstrum) The mel frequency cepstral coefficients (MFCCs) of a signal are a small set of features (usually about 10-20) which concisely describe the overall shape of a spectral envelope.
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0)

    # chroma
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)

    # melspectrogram It partitions the Hz scale into bins, and transforms each bin into a corresponding bin in the Mel Scale, using a overlapping triangular filters.
    mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)

    # spectral contrast Each frame of a spectrogram S is divided into sub-bands. For each sub-band, the energy contrast is estimated by comparing the mean energy in the top quantile (peak energy) to that of the bottom quantile (valley energy). High contrast values generally correspond to clear, narrow-band signals, while low contrast values correspond to broad-band noise.
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)

    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T,axis=0)
    return mfccs,chroma,mel,contrast,tonnetz


mfccs_lista = []
chroma_lista = []
mel_lista = []
contrast_lis = []
ton_lis = []

for i in new_x:
    mfccs,chroma,mel,contrast,tonnetz = feat_prova(i)
    mfccs_lista.append(mfccs)
    chroma_lista.append(chroma)
    mel_lista.append(mel)
    contrast_lis.append(contrast)
    ton_lis.append(tonnetz)

mfccs_list = np.array(mfccs_lista)
chroma_lista = np.array(chroma_lista)
mel_lista = np.array(mel_lista)
contrast_lis= np.array(contrast_lis)
ton_lis = np.array(ton_lis)

"""
import scipy.stats as stats
def get_features(frequencies):
    nobs, minmax, mean, variance, skew, kurtosis = stats.describe(frequencies)
    median    = np.median(frequencies)
    std       = np.std(frequencies)
    low,peak  = minmax
    q75,q25   = np.percentile(frequencies, [75 ,25])
    iqr       = q75 - q25
    return nobs, mean, skew, kurtosis, median, std, low, peak, q25, q75, iqr
nob_lis = []
mean_lis = []
ske_lis = []
kur_lis = []
median_lis = []
std_lis = []
low_lis = []
peak_lis = []
q25_lis = []
q75_lis = []
mode_lis = []
iqr_lis = []
for i in new_x:
    nobs, mean, skew, kurtosis, median, std, low, peak, q25, q75, iqr =get_features(i)
    nob_lis.append(nobs)
    mean_lis.append(mean)
    ske_lis.append(skew)
    kur_lis.append(kurtosis)
    median_lis.append(median)
    std_lis.append(std)
    low_lis.append(low)
    peak_lis.append(peak)
    q25_lis.append(q25)
    q75_lis.append(q75)
    iqr_lis.append(iqr)

mean_spec = np.mean(spec, axis = 1)
std_spec = np.std(spec,axis = 1)
zcr.shape
features = np.concatenate([np.array(mean_lis).reshape(150,1), np.array(ske_lis).reshape(450,1), np.array(kur_lis).reshape(450,1),
                           np.array(median_lis).reshape(450,1), np.array(std_lis).reshape(450,1), np.array(low_lis).reshape(450,1),
                           np.array(peak_lis).reshape(450,1)], axis =1)
"""

features2 = np.concatenate([mfccs_list,chroma_lista,mel_lista,contrast_lis, ton_lis], axis = 1)
#np.array(mean_lis).reshape(450,1),np.array(peak_lis).reshape(450,1), np.array(ske_lis).reshape(450,1),np.array(low_lis).reshape(450,1)


"""
NEW TRAIN set
"""
#X_tot = np.concatenate([mean_spec, std_spec, zcr.reshape(450, 216), rmse.reshape(450,216)], axis = 1)

"""
Preprocessing
"""
from sklearn.preprocessing import StandardScaler, LabelEncoder
sst = StandardScaler()
X_tot = sst.fit_transform(features2)

encoder = LabelEncoder()
y_cosa_dummy = encoder.fit_transform(new_y_cosa)

encoder2 = LabelEncoder()
y_chi_dummy = encoder2.fit_transform(new_y_chi)


"""
Dimensionality reduction
"""
from sklearn.utils import shuffle
X_train, y_train_chi = shuffle(X_tot, y_chi_dummy)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train_chi,
                                                    stratify=y_train_chi,
                                                test_size=0.3)

from sklearn.decomposition import PCA, KernelPCA
pca = PCA(0.95)

x_train_pca = pca.fit_transform(X_train)
x_test_pca = pca.transform(X_test)

"""
Machine Learning Classification Chi Parla?
"""
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.multiclass import OneVsOneClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process import GaussianProcessClassifier
#clf = SVC(kernel = "rbf", gamma = 'auto')#OneVsRestClassifier(SVC(C=13, kernel='rbf', gamma = 'auto'))
#clf = GradientBoostingClassifier(n_estimators=200, learning_rate=0.001)
#clf = RandomForestClassifier(n_estimators=100)
clf = MLPClassifier(random_state=0, hidden_layer_sizes=(200))
#clf = LogisticRegression()
scores = cross_val_score(clf, X_train, y_train, cv=3)
clf.fit(X_train, y_train)
test_pred = clf.predict(X_test)
score2 = accuracy_score(y_test,test_pred)
print(" accuracy test: ", score2)
print('Accuracy CV & variance: ', scores.mean(), scores.std() * 2)

from sklearn.metrics import classification_report
print(classification_report(test_pred, y_test))

import time
import sounddevice as sd
duration = 2.5 # seconds
fs = 44100

prova = sd.rec(int(duration * fs), samplerate=fs, channels=1)
sd.play(prova)
mfccs,chroma,mel,contrast,tonnetz = feat_prova(prova.reshape(110250))
nobs, mean, skew, kurtosis, median, std, low, peak, q25, q75, iqr = get_features(prova.reshape(110250))
x_prova = np.concatenate([mfccs.reshape(1,40),chroma.reshape(1,12),mel.reshape(1,128),contrast.reshape(1,7), tonnetz.reshape(1,6)],axis = 1)
x_prova = sst.transform(x_prova)
pred = clf.predict_proba(x_prova)
pred
encoder2.inverse_transform(pred.argmax(axis = -1))

"""
Machine Learning Classification Comando?
"""

from sklearn.utils import shuffle
X_train, y_train_cosa = shuffle(X_tot, y_cosa_dummy)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train_cosa,
                                                    stratify=y_train_cosa,
                                                test_size=0.3)

clf = MLPClassifier(random_state=0, hidden_layer_sizes=(150,))
#clf = LogisticRegression()
scores = cross_val_score(clf, X_train, y_train, cv=3)
clf.fit(X_train, y_train)
test_pred = clf.predict(X_test)
score2 = accuracy_score(y_test,test_pred)
print(" accuracy test: ", score2)
print('Accuracy CV & variance: ', scores.mean(), scores.std() * 2)
print(classification_report(test_pred, y_test))

prova = sd.rec(int(duration * fs), samplerate=fs, channels=1)
sd.play(prova)
mfccs,chroma,mel,contrast,tonnetz = feat_prova(prova.reshape(110250))
nobs, mean, skew, kurtosis, median, std, low, peak, q25, q75, iqr = get_features(prova.reshape(110250))
x_prova = np.concatenate([mfccs.reshape(1,40),chroma.reshape(1,12),mel.reshape(1,128),contrast.reshape(1,7), tonnetz.reshape(1,6)],axis = 1)
x_prova = sst.transform(x_prova)
pred = clf.predict_proba(x_prova)
np.set_printoptions(suppress=True)
pred
print("Hai detto la seguente parola: ",str(encoder.inverse_transform(pred.argmax(axis = -1))))


"""
Transfer Learning
"""

def preprocess_input(x):
    x /= 255.
    x -= 0.5
    x *= 2.
    return x



all_y = np.concatenate([new_y_chi.reshape(450,1), new_y_cosa.reshape(450,1)], axis = 1)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(spec, all_y,
                                                    stratify=all_y,
                                                    test_size=0.10)
from sklearn.utils import shuffle
X_train, y_train = shuffle(X_train,y_train,random_state = 1)
X_test, y_test = shuffle(X_test, y_test,random_state = 1)



X_train = preprocess_input(X_train)

def add_3d(x):
  l = []
  for i in x:#if the image is gray scale shape
      img2 = np.stack((i,i,i), axis = -1) #add three channels
      l.append(img2)
  return np.asarray(l)
X_train = add_3d(X_train)


from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
encoder.fit(y_train[:,0])
y_chi_dummy = encoder.transform(y_train[:,0])
y_chi_dummy = keras.utils.np_utils.to_categorical(y_chi_dummy)
encoder2 = LabelEncoder()
encoder2.fit(y_train[:,1])
y_cosa_dummy = encoder2.transform(y_train[:,1])
y_cosa_dummy = keras.utils.np_utils.to_categorical(y_cosa_dummy)

X_train.shape

"""
Model
"""
from keras.preprocessing import image as kimage
from keras.applications.inception_v3 import InceptionV3
model = InceptionV3(input_shape = (1025, 216, 3),
                 include_top=False)

from keras.models import Model
from keras.layers import Dense, Conv2D
from keras.layers import Input, Flatten, Dropout, MaxPooling2D
from keras import optimizers
sgd = optimizers.SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
for layer in model.layers:
  layer.trainable = False
#Build our NN classificator
x = model.output
x = Flatten()(x)
x = Dense(64, activation = "relu")(x)
pred_chi = Dense(2, activation = "sigmoid", name = 'chi_output')(x)
pred_cosa = Dense(3, activation = "softmax", name = 'cosa_output')(x)
net = Model(inputs=model.input, outputs = [pred_chi, pred_cosa])
net.compile(loss = [keras.losses.binary_crossentropy, keras.losses.categorical_crossentropy],
            optimizer = sgd, metrics = ['accuracy'])
net.fit(X_train,[y_chi_dummy, y_cosa_dummy],validation_split = 0.1, epochs = 10, batch_size=32)

net.evaluate(X_train, [y_chi_dummy, y_cosa_dummy])


mfcc.shape
spec.shape
