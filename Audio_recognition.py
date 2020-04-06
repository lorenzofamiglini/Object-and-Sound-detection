# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 21:48:52 2019

@author: loren
"""

import pandas as pd
import numpy as np
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

giox_path = "D:/Data_Science_all/MSC_2_anno/PROGETTO_AML_DSIM/Test_Audio/Giorgio_set/"
lox_path = "D:/Data_Science_all/MSC_2_anno/PROGETTO_AML_DSIM/Test_Audio/Lorenzo_set/"
#Descrizione + chi ha parlato Lorenzo
lox_descr = np.load(lox_path+"lor_audio_descrizione.npy")
lox_y_descr_de = np.repeat("descrizione", len(lox_descr))
lox_y_descr_chi = np.repeat("lorenzo", len(lox_descr))  

lox_descr2 = np.load(lox_path+"lor_audio_descrizione2.npy").reshape(10,110250)
lox_y_descr_de2 = np.repeat("descrizione", len(lox_descr2))
lox_y_descr_chi2 = np.repeat("lorenzo", len(lox_descr2))
#Sanzione + chi ha parlato Lorenzo
lox_sanz = np.load(lox_path+"lor_audio_sanzione.npy")
lox_y_sanz_sa = np.repeat("sanzione", len(lox_sanz))
lox_y_sanz_chi = np.repeat("lorenzo", len(lox_sanz))

lox_sanz2 = np.load(lox_path+"lor_audio_sanzione2.npy").reshape(10,110250)
lox_y_sanz_sa2 = np.repeat("sanzione", len(lox_sanz2))
lox_y_sanz_chi2 = np.repeat("lorenzo", len(lox_sanz2))
# Tipologia + chi ha parlato Lorenzo
lox_tipo = np.load(lox_path+"lor_audio_tipologia.npy")
lox_y_tipo_ti = np.repeat("tipologia", len(lox_tipo))
lox_y_tipo_chi = np.repeat("lorenzo", len(lox_tipo))

lox_tipo2 = np.load(lox_path+"lor_audio_tipologia2.npy").reshape(10,110250)
lox_y_tipo_ti2 = np.repeat("tipologia", len(lox_tipo2))
lox_y_tipo_chi2 = np.repeat("lorenzo", len(lox_tipo2))
#Descrizione + chi ha parlato Giorgio
gio_descr = np.load(giox_path+"giox_audio_descrizione.npy")
gio_y_descr_de = np.repeat("descrizione", len(gio_descr))
gio_y_descr_chi = np.repeat("giorgio", len(gio_descr))

gio_descr2 = np.load(giox_path+"giox_audio_descrizione2.npy").reshape(10,110250)
gio_y_descr_de2 = np.repeat("descrizione", len(gio_descr2))
gio_y_descr_chi2 = np.repeat("giorgio", len(gio_descr2))
#Sanzione + chi ha parlato Giorgio
gio_sanz = np.load(giox_path+"giox_audio_sanzione.npy")
gio_y_sanz_sa = np.repeat("sanzione", len(gio_sanz))
gio_y_sanz_chi = np.repeat("giorgio", len(gio_sanz))

gio_sanz2 = np.load(giox_path+"giox_audio_sanzione2.npy").reshape(10,110250)
gio_y_sanz_sa2 = np.repeat("sanzione", len(gio_sanz2))
gio_y_sanz_chi2 = np.repeat("giorgio", len(gio_sanz2))
# Tipologia + chi ha parlato Giorgio
gio_tipo = np.load(giox_path+"giox_audio_tipologia.npy")
gio_y_tipo_ti = np.repeat("tipologia", len(gio_tipo))
gio_y_tipo_chi = np.repeat("giorgio", len(gio_tipo))

gio_tipo2 = np.load(giox_path+"giox_audio_tipologia2.npy").reshape(10,110250)
gio_y_tipo_ti2 = np.repeat("tipologia", len(gio_tipo2))
gio_y_tipo_chi2 = np.repeat("giorgio", len(gio_tipo2))

"""
Uniamo tutti gli audio con le giuste labels y
"""

X = np.concatenate([lox_descr,lox_descr2, gio_descr,gio_descr2, lox_sanz,lox_sanz2,
                    gio_sanz,gio_sanz2, lox_tipo,lox_tipo2, gio_tipo,gio_tipo2], axis = 0)
y_cosa = np.concatenate([lox_y_descr_de,lox_y_descr_de2, gio_y_descr_de, gio_y_descr_de2, lox_y_sanz_sa,lox_y_sanz_sa2,
                         gio_y_sanz_sa,gio_y_sanz_sa2,lox_y_tipo_ti,lox_y_tipo_ti2, gio_y_tipo_ti,gio_y_tipo_ti2], axis = 0)
y_chi = np.concatenate([lox_y_descr_chi,lox_y_descr_chi2, gio_y_descr_chi, gio_y_descr_chi2,lox_y_sanz_chi, lox_y_sanz_chi2,
                        gio_y_sanz_chi,gio_y_sanz_chi2, lox_y_tipo_chi, lox_y_tipo_chi2,gio_y_tipo_chi,gio_y_tipo_chi2], axis = 0)


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

import librosa

def feat_prova(X):
    sample_rate = 44100
    # short term fourier transform: represents a signal in the time-frequency domain by computing discrete Fourier transforms (DFT) over short overlapping windows.
    stft = np.abs(librosa.stft(X))

    # mfcc (mel-frequency cepstrum) The mel frequency cepstral coefficients (MFCCs) of a signal are a small set of features (usually about 10-20) which concisely describe the overall shape of a spectral envelope.
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0)

    # chroma, scarta il timbro da rivedere 
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

#Plottiamo le varie trasformazioni, la prima è il suono originale
plt.figure(figsize=(15,15))
plt.subplot(131)
plt.plot(X[0])
plt.subplot(133)
plt.plot(mfccs_list[0])
plt.subplot(133)
plt.plot(chroma_lista[0])
plt.subplot(133)
plt.plot(mel_lista[0])
plt.subplot(133)
plt.plot(contrast_lis[0])
plt.subplot(133)
plt.plot(ton_lis[0])
plt.ylim(-100,100)

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


import time

import sounddevice as sd

"""
Conversazione Task
"""

df = pd.read_csv("D:/Data_Science_all/MSC_2_anno/PROGETTO_AML_DSIM/Object-and-Sound-detection/risposte_domande_trafficlight.csv", sep = ";")
df.drop("Unnamed: 0", axis=1, inplace=True)
df.columns = ['title', 'target', 'descrizione', 'tipologia', 'sanzione']
df
from gtts import gTTS
from playsound import playsound

def question_answer(segnale, audio):
    #Preprocessing audio di input
    mfccs,chroma,mel,contrast,tonnetz = feat_prova(audio.reshape(110250))
    x_prova = np.concatenate([mfccs.reshape(1,40),chroma.reshape(1,12),mel.reshape(1,128),
                              contrast.reshape(1,7), tonnetz.reshape(1,6)],axis = 1)
    x_prova = sst.fit_transform(x_prova)

    chi = ''.join(encoder2.inverse_transform(clf_chi.predict(x_prova)))
    cosa = ''.join(encoder.inverse_transform(clf.predict(x_prova)))


    risposta = df[df["title"]==segnale][cosa].iloc[0]

    if (cosa == "descrizione"):
        a = str("Ciao "+chi+", "+"il segnale di: "+segnale+", e' descritto nel seguente modo: "+risposta)
    if (cosa == "sanzione"):
        a = str("Ciao "+chi+", "+"il segnale di: "+segnale+", presenta la seguente sanzione: "+risposta)
    if (cosa == "tipologia"):
        a = str("Ciao "+chi+", "+"il segnale di: "+segnale+", rientra nella tipologia di: "+risposta)
    language = "it"
    text = a
    myobj = gTTS(text=text, lang=language, slow=False)
    myobj.save("D:/Data_Science_all/MSC_2_anno/PROGETTO_AML_DSIM/Object-and-Sound-detection/audio.mp3")

    return a, playsound("D:/Data_Science_all/MSC_2_anno/PROGETTO_AML_DSIM/Object-and-Sound-detection/audio.mp3")
audio_z = sd.rec(int(duration * fs), samplerate=fs, channels=1)
app = question_answer(segnale = 'Pericolo ghiaccio', audio = audio_z)
sklearn.__version__

from joblib import dump, load

clf = load('D:/Data_Science_all/MSC_2_anno/PROGETTO_AML_DSIM/modello_cosa.joblib') 
clf_chi = load('D:/Data_Science_all/MSC_2_anno/PROGETTO_AML_DSIM/modello_chi.joblib') 
#importiamo il modello
#clf = load('filename.joblib') 