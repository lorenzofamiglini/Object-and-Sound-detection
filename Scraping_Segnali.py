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

link = "http://quizscuolaguida.altervista.org/teoria/indicazione+.html" #take the general link
for i in range(4,7):
    link = re.sub(r'(\+)',str(i), link) #we need to replace "+" and insert the title
    try:
        result = requests.get(link, headers=headers)
        soup = BeautifulSoup(result.content.decode(), 'html.parser')
        descrizione, titolo = get_tile_description(soup)
        all_description.extend(descrizione)
        all_title.extend(titolo)
    except AttributeError:
        pass
    link = "http://quizscuolaguida.altervista.org/teoria/indicazione+.html"

#pericolo, precedenza, divieto, obbligo, indicazione (pagine 4-6),
all_description
all_title

df = pd.DataFrame()
df["title"] = all_title
df["description"] = all_description

df


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
