# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 22:40:51 2019

@author: loren

laura
Ginevra
Isabella
Arianna
Massimo
Papa giorgio
Riccardo

"""
import numpy as np
import sounddevice as sd
import pandas as pd
duration = 2.5 # seconds
fs = 44100

#Persone che dicono tipologia 
#persone_tipologia = np.zeros(110250).reshape(110250,1)

registrazione = sd.rec(int(duration * fs), samplerate=fs, channels=1)
persone_tipologia = np.concatenate([ persone_tipologia, registrazione], axis = 1) #4 maschi e 4 femmine per 9 volte = 72 
sd.play(persone_tipologia[:,38])

persone_descrizione[:,1].shape
#Persone che dicono descrizione 
#persone_descrizione = np.zeros(110250).reshape(110250,1)

registrazione = sd.rec(int(duration * fs), samplerate=fs, channels=1)
persone_descrizione = np.concatenate([persone_descrizione, registrazione], axis = 1)
 #4 maschi e 4 femmine per 9 volte = 72 audio 
sd.play(persone_sanzione[:,37])

#Persone che dicono sanzione 
persone_sanzione = np.zeros(110250).reshape(110250,1)

registrazione = sd.rec(int(duration * fs), samplerate=fs, channels=1)
persone_sanzione = np.concatenate([persone_sanzione, registrazione], axis = 1) #4 maschi e 4 femmine per 9 volte = 72 audio 

sd.play(persone_sanzione[:,])

#Salvare array audio 

#np.save("D:/Data_Science_all/MSC_2_anno/PROGETTO_AML_DSIM/Test_Audio/AltreVoci_set/persone_tipologia", persone_tipologia)
#np.save("D:/Data_Science_all/MSC_2_anno/PROGETTO_AML_DSIM/Test_Audio/AltreVoci_set/persone_descrizione", persone_descrizione)
#np.save("D:/Data_Science_all/MSC_2_anno/PROGETTO_AML_DSIM/Test_Audio/AltreVoci_set/persone_sanzione", persone_sanzione)


#Check se tutto è stato salvato correttamente 
altrev_path = "D:/Data_Science_all/MSC_2_anno/PROGETTO_AML_DSIM/Test_Audio/AltreVoci_set/"
persone_tipologia2 = np.load(altrev_path+"persone_tipologia.npy")
persone_descrizione2 = np.load(altrev_path+"persone_descrizione.npy")
persone_sanzione2 = np.load(altrev_path+"persone_sanzione.npy")

sd.play(persone_sanzione[:,28]) #da 28 a 36 donna 
sd.play(persone_descrizione2[:,38]) #fino a 38 donna 
sd.play(persone_tipologia2[:,39]) #fino a 38 donna 


persone_descrizione2.shape


