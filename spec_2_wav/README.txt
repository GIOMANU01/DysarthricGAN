la cartella Spec_2_wav contiene:
--Vocoding.ipynb che è costituito da due parti:
   1 carica la struttura del generatore della GAN e i pesi del miglior generatore post addestramento e genera i dati sintetici a partire dai sani stretchati
   2 carica Waveglow ed esegue la conversione da spettrogramma ad audio degli spettrogrammi del punto 1

-- dal github https://github.com/NVIDIA/waveglow/ scaricare (sono presenti anche in questa cartella):
-- denoiser.py (si applica dopo waveglow
-- stft.py, layers.py e audio_processing.py che servono per il denoiser.py e si trovano dentro la cartella tacotron2
