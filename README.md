# Object-and-Sound-detection traffic sign and word trigger recognition: 

Object and Sound detection with deep learning and Machine Learning techniques. For the optimization part, AutoML was applied in order to obtain good models. 

Ogni file è di seguito descritto:

- DataAugmentationForObjectDetection-master: contiene script utili per la Data Augmentation per l'object Detection
					     (fonte: https://blog.paperspace.com/data-augmentation-for-bounding-boxes/)

- CNN - sliding window: implementazione convoluzionale dell'algoritmo di sliding window. 

- Multi-task model & SigOpt opt: creazione del modello CNN multi-task con implementazione di tecniche di ottimizzazione tramite la libreria SigOpt.

- Cluster Analysis: Feature extraction sul dataset handcrafted e analisi dei cluster per l'interpretazione dell'apprendimento della rete.
		    Vi sono anche le visualizzazioni delle feature map della rete per i vari layer.

- Collegamento tra le reti: Script che permette di collegare l'output della rete CNN - sliding window all'input della rete multi-task. Vi sono anche funzioni utili per la demo live. 

- Import dei dati: codice che permette di importare le immagini dalle cartelle per poi esportare i dati in formato numpy array. 

- Progetto_audio_segnali_automl: Sviluppo dei modelli (MLP per il riconoscimento dell'identità e del contenuto dei file audio) e ottimizzazione degli stessi con AutoML. 

- rec_voci: registrazione dei comandi vocali.

- Audio_recognition: Feature Extraction e Feature Selection per i file audio.

- Demo_Riconoscimento_Vocale: script che implementa funzioni utili per il processing di file audio per la demo live.

- Demo Live - Immagini: script che implementa funzioni utili per la demo live nel processing di immagini.

- Scraping_Segnali: script che permette di effettuare lo scraping dal sito http://quizscuolaguida.altervista.org/.

- Transfer_learning_segnali_stradali: è stato utilizzato per poter applicare il Transfer Learning per il task di classificazione di presenza o non presenza del segnale stradale all'interno di un immagine.

- Classificazione_multiclasse_transfer_learning: è stato usato per risolvere il task di classificazione del segnale e di bounding box, mediante l'utilizzo del transfer learning.

I dataset utilizzati sono disponibili on-line ai seguenti link:

- http://benchmark.ini.rub.de/?section=gtsrb&subsection=news

- https://www.crcv.ucf.edu/data/GMCP_Geolocalization/

- https://www.kaggle.com/arnaud58/landscape-pictures

- https://forums.fast.ai/t/detecting-coconut-trees-from-the-air-with-fast-ai-notebooks-dataset-available/14194
