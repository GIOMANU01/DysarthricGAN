il dataset di riferimento è stato scaricato dal sito https://www.kaggle.com/datasets/aryashah2k/noise-reduced-uaspeech-dysarthria-dataset

il preprocessing segue queste fasi
FASE 1 
a)dataset_analysis_control.ipynb per determinare il SNR delle tracce nei diversi canali dei control
b)dataset_analysis_patient.ipynb per determinare il SNR delle tracce nei diversi canali dei patient

FASE 2
dataset_best_SNR.ipynb sceglie le tracce per ogni parola con il miglio SNR (da fare sia per control che per patient

FASE 3
a) applicare VAD_silero.ipynb per rimuovere silenzi ad inizio e fine delle tracce: 
--- in questa fase sono stati eliminati i file con audio corrotto 
--- gli audio in cui il vad creava più di un segmento sono stati tagliati manualmente mantenendo la o le pause intermedie
--- gli audio che il vad non riusciva a modificare sono stati tagliati manualmente
b) rename.py rinomina i file eliminando "_segment1"

FASE 4
scegliere le coppie sano-disartrico a cui sottoporre l'allineamento e il successivo training GAN

FASE 5 
a) "create -n aligner -c conda-forge montreal-forced-aligner" crea un ambiente conda dove installa in automatico al suo interno il Montreal Forced Aligner (MFA)
b) no_word_detection.ipynb serve per verificare che le parole del dataset siano contenute nel vocabolario english_mfa.dict 
---le parole non presenti sono state aggiunte manualmente nel .dict con la relativa fonetica. (per i simboli fonetici fare riferimento al file phones.txt
che si trova in acoustic/english_mfa/english_mfa cartella che viene scaricato quando si avvia il comando del punto a). 
c) applicare mfa_aligner.py che elimina i silenzi pre e post ( se sono rimasti dopo il vad), allinea forzatamente i fonemi all'audio e restituisce la differenza di tempo
della parola detta dal sano e dal disartrico

FASE 6 
applicare WSOLA_Speed_alg.ipynb (ambiente python 3.10.9) per modificare la durata dei sani in modo che vengono allungati o accorciati alla durata dei disartrici 
ci sono due modi per modificare la durata:
--- WSOLA permette di modificare la durata del sano senza modificarne il pitch
--- speed modifica la durata del sano modificanod anche il pitch

CLEESE implemente l'algoritmo WSOLA con prestazioni di stretching ottime
per usare CLEESE i passaggi sono:
1) creare un ambiente su anaconda o shell nominato cleese con python 3.10
2) scaricare cleese tramite "pip install cleese-stim" oppure "pip install git+https://github.com/neuro-team-femto/cleese.git"
3) installare le librerie necessarie per far funzionare il codice che si vuole far girare usando cleese
4) nel codice importare cleese cosi "import cleese_stim as cleese"
5) import toml, from cleese_stim.engines.phase_vocoder.phase_vocoder import PhaseVocoder servono per fare time stretching dei file audio.
6) dal https://github.com/neuro-team-femto/cleese.git scaricare il file docs/api/configs/cleese-phase-vocoder.toml 
7) dentro tale file aggiungere param_ext = ".txt" dopo riga 16, 
   modificare num_files = 10 --> num_files = 1, 
   transf = ["stretch", "pitch", "eq", "gain"] -->  transf = ["stretch"]
8) a questo punto nel codice principale per fare time stretching usare process.phaseVocoder e definire i parametri


FASE 7
a) determinare la coppia di parole più lunga (es 1.25 s è la parola più lunga
b) norm_track.ipynb (ambiente python 3.10.9) applica le seguenti modifiche:
--- zeropadding su tutti i file wav fino a raggiungere la lunghezza del max (es 1.25 s), nel fare questo applica un fade_out di 20 ms per evitare componenti ad alta frequenza
--- normalizza ogi traccia rispetto al suo massimo
--- crea gli spettrogrammi in formato mel a 80 dimensioni (in formato .pth)
--- normalizza gli spettrogrammi rispetto alla media e varianza dello speaker
norm_track.ipynb è stato modificato in modo che il dataset non conenga solo sano.pth e disartrico.pth ma anche il sano_in.pth che sarebbe il sano prima che venga applicato lo stretching 


A QUESTO PUNTO VIENE CREATO UN DATASET DI QUESTO TIPO
MXX_dataset----> B1_C1---->sano.pth                               
           |          ---->disartrico.pth  dove XX indica il paziente (14, 04, 07 ecc...)
           |          ---->sano_in.pth                                                          B1_C1 indica il codice riferito alla parola detta   
           |                                                                                    sano.pth, sano_in.pth edisartrico.pth indicano gli spettrogrammi 
           |---> B1_C2---->sano.pth 
                      ---->disartrico.pth
              .       ---->sano_in.pth
              .
              .


               




