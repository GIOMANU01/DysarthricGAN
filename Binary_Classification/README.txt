la cartella Bynary_Classifiation contiene:
--data_preprocess.py: 
    1. determino il valore ottimo per pad/trim (224 frames ovvero 2.61 s alla Fs = 22050 Hz)
    2. applico il pad/trim ai file audio originali e generati 
    3. genero gli spettrogrammi originali e generati (lineari 224*224)
    4. genero il file csv con tutte le caratterisitche di ogni spettrogramma
    5. calcolo media e varianza globali e per i soli originali per la z-score norm

-- config_dys.py: contiene i parametri di configurazione, tra i più importanti:
    --train, val, test sets
    --cfg.augm = TRUE se voglio includere nel training i dati generati dalla GAN, FALSE altrimenti 
    -- media e varianza per la normalizzazione z-score 

--dataset.py:
    --carica lo spettrogramma con torch.load
    --assegna la label allo spettrogramma (0 se intell_level <= 50, 1 altrimenti)
    --defnisce la z-score norm

--main_binary_classification.py: carica la ResNEt-50 per la classifcazione e attiva l'addestramento
    
