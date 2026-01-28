la cartella Codici_comuni contiene tutti i codici che sono identici per qualsiasi tipo di paziente:
--DCGAN_dys.py scodice dell'architettura del G e del D 
--utils_GAN.py contiene alcune funzioni utilizzate nei codici come quella per importare il dataset 
--losses.py contiene alcune loss utilizzate nel training 
--train_op.py  è il codice utilizzato per il training dei trial di optuna (gestisce i centroidi in validation per scegliere il miglior modello con batch 1) (da usare con i optuna_run_A00.py)
--train_cl.py  serve per fare il training senza validation e test per la successiva classificazione dando alla rete tutte le parole dell'UA_speech per paziente (gestisce i centroidi considerando non piu un batch 1 ma 32)
  (da usare con i run_cl_A00.py)

