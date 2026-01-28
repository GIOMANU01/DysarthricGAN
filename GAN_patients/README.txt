la cartella GAN_patients contiene 2 cartelle:
-- Metodo Automatico:
    -- valutare la generazione del miglior modello di optuna per ogni singolo paziente tramite lo studio dei centroidi
    --si calcolano 3 distanze:
      --D1: rappresneta la distanza tra i disartrici del UA-Speech e generati dalla rete (più è piccola meglio è)
      --D2: rappresenta la distanza tra i sani dell'UA-Speech e i generati (più è simile a D3 meglio è) 
      --D3: rappresenta la distanza tra il sano e il disartrico dell'UA_speech

--Metodo Manuale:
   si valuta il miglior chechpoint del generatore in base a 5 file audio salvati ogni 15 epoche. si prende il generatore che ascoltando gli audio li genera di buona qualità e con caratterisithce disartriche percepite all'ascolto.
