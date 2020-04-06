import requests
import urllib.request
import time
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import re

"""
Web scraping delle informazione dei segnali stradali
"""
url = 'http://quizscuolaguida.altervista.org/teoria/pericolo1.html'
headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/73.0.3657.0 Safari/537.36'}
result = requests.get(url, headers=headers)
print(result.content.decode())


def get_tile_description(soup):
    descrizione = []
    titolo = []
    descrizione = [descrizione.text.strip() for descrizione in soup.findAll("div", {"class" : "contenuto" })]
    titolo = [titolo.text.strip() for titolo in soup.findAll("div", {"class" : "titolo" })]
    return descrizione, titolo

all_description = []
all_title = []
voci = ["pericolo","precedenza", "divieto"]

link = "http://quizscuolaguida.altervista.org/teoria/complementari+.html" #take the general link
for i in range(1,5):
    link = re.sub(r'(\+)',str(i), link) #we need to replace "+" and insert the title
    try:
        result = requests.get(link, headers=headers)
        soup = BeautifulSoup(result.content.decode(), 'html.parser')
        descrizione, titolo = get_tile_description(soup)
        all_description.extend(descrizione)
        all_title.extend(titolo)
    except AttributeError:
        pass
    link = "http://quizscuolaguida.altervista.org/teoria/complementari+.html"

"""
Creazione del dataframe con il target, titolo e descrizione
"""
#pericolo, precedenza, divieto, obbligo, indicazione, complementari
len(all_description)
len(all_title)
agggiornamento = ['Limite massimo di velocità di 20 Km/h',
'Limite massimo di velocità di 30 Km/h',
'Limite massimo di velocità di 50 Km/h',
'Limite massimo di velocità di 60 Km/h',
'Limite massimo di velocità di 70 Km/h',
'Limite massimo di velocità di 100 Km/h',
'Limite massimo di velocità di 120 Km/h']
aggiornamento_descr = ["Indica la velocità massima consentita per tutti i veicoli, ha validità immediatamente dopo il segnale; prescrive di marciare a velocità inferiore o uguale al valore indicato. Bisogna tenere conto però delle condizioni della strada (tale segnale non indica che in ogni circostanza, anche con scarsa visibilità, possiamo procedere a 20 km/h). Il segnale non indica la velocità consigliata o quella minima, ma la velocità massima da non superare. Vale anche per i motocicli. Si può trovare su strade extraurbane ed in autostrada (ad esempio: quando ci sono lavori in corso).",
                       "Indica la velocità massima consentita per tutti i veicoli, ha validità immediatamente dopo il segnale; prescrive di marciare a velocità inferiore o uguale al valore indicato. Bisogna tenere conto però delle condizioni della strada (tale segnale non indica che in ogni circostanza, anche con scarsa visibilità, possiamo procedere a 30 km/h). Il segnale non indica la velocità consigliata o quella minima, ma la velocità massima da non superare. Vale anche per i motocicli. Si può trovare su strade extraurbane ed in autostrada (ad esempio: quando ci sono lavori in corso).",
                       "Indica la velocità massima consentita per tutti i veicoli, ha validità immediatamente dopo il segnale; prescrive di marciare a velocità inferiore o uguale al valore indicato. Bisogna tenere conto però delle condizioni della strada (tale segnale non indica che in ogni circostanza, anche con scarsa visibilità, possiamo procedere a 50 km/h). Il segnale non indica la velocità consigliata o quella minima, ma la velocità massima da non superare. Vale anche per i motocicli. Si può trovare su strade extraurbane ed in autostrada (ad esempio: quando ci sono lavori in corso).",
                       "Indica la velocità massima consentita per tutti i veicoli, ha validità immediatamente dopo il segnale; prescrive di marciare a velocità inferiore o uguale al valore indicato. Bisogna tenere conto però delle condizioni della strada (tale segnale non indica che in ogni circostanza, anche con scarsa visibilità, possiamo procedere a 60 km/h). Il segnale non indica la velocità consigliata o quella minima, ma la velocità massima da non superare. Vale anche per i motocicli. Si può trovare su strade extraurbane ed in autostrada (ad esempio: quando ci sono lavori in corso).",
                       "Indica la velocità massima consentita per tutti i veicoli, ha validità immediatamente dopo il segnale; prescrive di marciare a velocità inferiore o uguale al valore indicato. Bisogna tenere conto però delle condizioni della strada (tale segnale non indica che in ogni circostanza, anche con scarsa visibilità, possiamo procedere a 70 km/h). Il segnale non indica la velocità consigliata o quella minima, ma la velocità massima da non superare. Vale anche per i motocicli. Si può trovare su strade extraurbane ed in autostrada (ad esempio: quando ci sono lavori in corso).",
                       "Indica la velocità massima consentita per tutti i veicoli, ha validità immediatamente dopo il segnale; prescrive di marciare a velocità inferiore o uguale al valore indicato. Bisogna tenere conto però delle condizioni della strada (tale segnale non indica che in ogni circostanza, anche con scarsa visibilità, possiamo procedere a 100 km/h). Il segnale non indica la velocità consigliata o quella minima, ma la velocità massima da non superare. Vale anche per i motocicli. Si può trovare su strade extraurbane ed in autostrada (ad esempio: quando ci sono lavori in corso).",
                       "Indica la velocità massima consentita per tutti i veicoli, ha validità immediatamente dopo il segnale; prescrive di marciare a velocità inferiore o uguale al valore indicato. Bisogna tenere conto però delle condizioni della strada (tale segnale non indica che in ogni circostanza, anche con scarsa visibilità, possiamo procedere a 120 km/h). Il segnale non indica la velocità consigliata o quella minima, ma la velocità massima da non superare. Vale anche per i motocicli. Si può trovare su strade extraurbane ed in autostrada (ad esempio: quando ci sono lavori in corso)."]
all_title.extend(agggiornamento)
all_description.extend(aggiornamento_descr)

df = pd.DataFrame()
df["title"] = all_title
df["description"] = all_description

"""
Match dei due dataframe per avere la variabile target associata
"""

match_diz = {"0":'Limite massimo di velocità di 20 Km/h',
"1":'Limite massimo di velocità di 30 Km/h',
"2":'Limite massimo di velocità di 50 Km/h',
"3":'Limite massimo di velocità di 60 Km/h',
"4":'Limite massimo di velocità di 70 Km/h',
"5":'Limite massimo di velocità di 80 Km/h',
"6":'Fine del limite massimo di velocità',
"7":'Limite massimo di velocità di 100 Km/h',
"8":'Limite massimo di velocità di 120 Km/h',
"9":'Divieto di sorpasso',
"10":'Divieto di sorpasso per gli autocarri che superano 3,5 T',
"11":'Inserzione con diritto di precedenza',
"12":'Diritto di precendenza',
"13":'Dare precedenza',
"14":'Fermarsi e dare precedenza (STOP)',
"15":'Divieto di transito',
"16":'Divieto di transito agli autocarri che superano 3,5 T',
"17":'Senso vietato',
"18":'Altri pericoli',
"19":'Curva a sinistra',
"20":'Curva a destra',
"21":'Doppia curva, la prima a sinistra',
"22":'Strada deformata',
"23":'Strada sdrucciolevole',
"24":'Strettoia asimmetrica a destra',
"25":'Lavori',
"26":'Preavviso di semaforo verticale',
"27":'Attraversamento pedonale',
"28":'Attenzione ai bambini',
"29":'Attraversamento ciclabile',
"30":'***',
"31":'Animali selvatici vaganti',
"32":'Via libera',
"33":'Preavviso direzione obbligatoria a destra',
"34":'Preavviso direzione obbligatoria a sinistra',
"35":'Direzione obbligatoria diritto',
"36":'Direzioni consentite diritto e destra',
"37":'Direzioni consentite diritto e sinistra',
"38":'Passaggio obbligatorio a destra',
"39":'Passaggio obbligatorio a sinistra',
"40":'Rotatoria',
"41":'Fine del divieto di sorpasso',
"42":'Fine del divieto di sorpasso per gli autocarri che superano 3,5 T'}
match_df = pd.DataFrame.from_dict(match_diz, orient = "index", columns = ["title"])
match_df["target"] = match_df.index

final_match = pd.merge(match_df, df, on='title', how='left', validate="one_to_many")

#final_match.to_csv("/Users/lorenzofamiglini/Desktop/final_segnali_stradali.csv")

final_match

"""
Salvare immagine dall'html
"""
link = "http://quizscuolaguida.altervista.org/teoria/pericolo1.html"
result = requests.get(link, headers=headers)
soup = BeautifulSoup(result.content.decode(), 'html.parser')
from skimage import io
import requests
from requests.exceptions import HTTPError
import numpy as np
import skimage
img = np.array([])
import scipy.misc
url = "http://quizscuolaguida.altervista.org/segnali/+.gif"
import urllib
conta = 54

for i in range(54,90):
    if (conta == 86):
        conta = 148
        url = re.sub(r'(\+)',str(conta), url)
    if conta == 148:
        conta = 91
        print(conta)
        url = re.sub(r'(\+)',str(conta), url)
    print("second"+str(conta))
    url = re.sub(r'(\+)',str(conta), url)
    image = io.imread(url)
    print(i)
    scipy.misc.imsave("/Users/lorenzofamiglini/Desktop/MsC_2_anno/PROGETTO_AML_DSIM/Segnali_Divieto/"+str(conta)+".png", image)
    conta = conta + 1


    url = "http://quizscuolaguida.altervista.org/segnali/+.gif"

all_description
